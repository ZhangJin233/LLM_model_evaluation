# app.py
import json, time, uuid
import streamlit as st
import pandas as pd
from datetime import datetime
import utils as U
import config as cfg

st.set_page_config(page_title="Ground-Truth Labeler", layout="wide")
st.title("ClickHouse Network-Request Ground Truth 标注")

###############################################################################
#   侧边栏：输入查询条件
###############################################################################
with st.sidebar:
    st.header("查询条件")
    cust_id = st.text_input("Customer ID", "")
    col1, col2 = st.columns(2)
    with col1:
        t0 = st.datetime_input(
            "Start Time", value=datetime.now().replace(hour=0, minute=0, second=0)
        )
    with col2:
        t1 = st.datetime_input("End Time", value=datetime.now())
    start_ev = st.text_input("Start Event (UI)", "")
    end_ev = st.text_input("End Event (UI)", "")
    if st.button("查询 ClickHouse"):
        if not (cust_id and start_ev and end_ev):
            st.error("Customer ID / Start Event / End Event 必填")
            st.stop()

        with st.spinner("正在查询 ClickHouse…"):
            raw_df = U.query_events(cust_id, t0, t1)
        if raw_df.empty:
            st.warning("找不到任何记录")
            st.stop()

        sliced_df = U.cut_between_events(raw_df, start_ev, end_ev)
        if sliced_df.empty:
            st.warning("所选事件范围内无数据 / 事件名不匹配")
            st.stop()

        req_df = U.filter_requests(sliced_df)
        if req_df.empty:
            st.warning("没有符合过滤条件的网络请求")
            st.stop()

        # 在 session_state 保存，方便后面按钮用
        st.session_state["req_df"] = req_df
        st.success(f"已捕获 {len(req_df)} 条网络请求")

###############################################################################
#   主区：展示结果 + 标注
###############################################################################
if "req_df" not in st.session_state:
    st.info("在侧边栏输入完条件并点击【查询】后，结果会显示在这里")
    st.stop()

req_df = st.session_state["req_df"].copy()
req_df["✅GroundTruth"] = False  # 添加一列可编辑布尔列

st.subheader("候选网络请求")
st.write("勾选 **✅GroundTruth** 列即可把该行标为真值。")
edited_df = st.data_editor(
    req_df,
    num_rows="dynamic",
    column_config={
        "✅GroundTruth": st.column_config.CheckboxColumn("✅GroundTruth", default=False)
    },
    height=400,
    use_container_width=True,
)

###############################################################################
#   完成按钮 → 生成结果 JSON + GCS
###############################################################################
if st.button("完成标注并生成 Ground Truth"):
    gt_paths = edited_df.loc[edited_df["✅GroundTruth"], "url"].tolist()
    if not gt_paths:
        st.warning("你还没勾任何行哦")
        st.stop()

    result = {
        "customer_id": cust_id,
        "start_event": start_ev,
        "end_event": end_ev,
        "ground_truth": gt_paths,
        "generated_at": int(time.time()),
    }
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    st.json(result)

    # 本地下载
    st.download_button(
        "下载 JSON",
        data=json_str,
        file_name=f"gt_{cust_id}_{uuid.uuid4().hex[:6]}.json",
        mime="application/json",
    )

    # 上传 GCS
    fname = f"gt_{cust_id}_{int(time.time())}.json"
    if U.upload_gcs(json_str, fname):
        st.success(f"已上传至 GCS：{cfg.GCS_BUCKET}/{fname}")
    else:
        st.error("上传 GCS 失败，请检查凭证/网络")
