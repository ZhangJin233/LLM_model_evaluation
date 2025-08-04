# config.py
# ——只放改一次就能全局生效的常量——

######################################################################
# ClickHouse 连接
######################################################################
CK_HOST = "your-clickhouse-host"
CK_PORT = 9440  # 如果用 HTTPS / TLS
CK_USER = "your_user"
CK_PASSWORD = "your_password"
CK_DATABASE = "your_db"

# 表名与字段（按你的库实际改）
CK_TABLE = "network_events"
# ↓下面这几个字段名要跟库里列名保持一致
COL_TIMESTAMP = "timestamp"
COL_EVENT_NAME = "event_name"
COL_CUSTOMER_ID = "customer_id"
COL_URL = "url"
COL_METHOD = "method"
COL_STATUS = "status_code"

######################################################################
# 要排除的静态资源 / 域名
######################################################################
EXCLUDE_EXT = [
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".css",
    ".js",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".map",
]
EXCLUDE_DOMAIN_KEYWORDS = [
    "google-analytics.com",
    "doubleclick.net",
]

######################################################################
# GCS 上传
######################################################################
GCS_BUCKET = "your-bucket"
GCS_CREDENTIAL_JSON = None  # 用默认 ADC 就写 None
