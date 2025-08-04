# utils.py
from datetime import datetime
from urllib.parse import urlparse
import json
import clickhouse_connect
import pandas as pd
from google.cloud import storage
import config as cfg


def ck_client():
    return clickhouse_connect.get_client(
        host=cfg.CK_HOST,
        port=cfg.CK_PORT,
        username=cfg.CK_USER,
        password=cfg.CK_PASSWORD,
        database=cfg.CK_DATABASE,
        secure=cfg.CK_PORT == 9440,  # 简单判断
    )


def query_events(
    customer_id: str, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    sql = f"""
        SELECT
            {cfg.COL_TIMESTAMP}  AS ts,
            {cfg.COL_EVENT_NAME} AS ev_name,
            {cfg.COL_URL}        AS url,
            {cfg.COL_METHOD}     AS method,
            {cfg.COL_STATUS}     AS status
        FROM {cfg.CK_TABLE}
        WHERE {cfg.COL_CUSTOMER_ID} = %(cid)s
          AND {cfg.COL_TIMESTAMP} >= %(t0)s
          AND {cfg.COL_TIMESTAMP} <= %(t1)s
        ORDER BY {cfg.COL_TIMESTAMP}
    """
    client = ck_client()
    df = client.query_df(
        sql, parameters={"cid": customer_id, "t0": start_time, "t1": end_time}
    )
    return df


def cut_between_events(df: pd.DataFrame, start_event: str, end_event: str):
    """在时间过滤后的 df 中，再按事件名裁剪 start_event→end_event 区段"""
    start_idx = df.index[df["ev_name"] == start_event]
    end_idx = df.index[df["ev_name"] == end_event]
    if len(start_idx) == 0 or len(end_idx) == 0:
        return pd.DataFrame()  # 找不到任一事件
    start_i = start_idx[0]
    end_i = end_idx[max(0, len(end_idx) - 1)]  # 取最后一个 end_event
    if start_i >= end_i:
        return pd.DataFrame()
    return df.loc[start_i:end_i]


def is_valid_req(url: str) -> bool:
    p = urlparse(url)
    if any(p.path.lower().endswith(ext) for ext in cfg.EXCLUDE_EXT):
        return False
    if any(dom in p.netloc.lower() for dom in cfg.EXCLUDE_DOMAIN_KEYWORDS):
        return False
    return True


def filter_requests(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["url"].apply(is_valid_req)
    return df[mask].reset_index(drop=True)


def upload_gcs(json_str: str, filename: str) -> bool:
    try:
        client = (
            storage.Client.from_service_account_json(cfg.GCS_CREDENTIAL_JSON)
            if cfg.GCS_CREDENTIAL_JSON
            else storage.Client()
        )
        bucket = client.bucket(cfg.GCS_BUCKET)
        blob = bucket.blob(filename)
        blob.upload_from_string(json_str, content_type="application/json")
        return True
    except Exception as e:
        print("GCS 上传失败→", e)
        return False
