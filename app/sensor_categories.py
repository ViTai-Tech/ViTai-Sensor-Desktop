# coding=utf-8
"""传感器类别定义与 SN 归类工具。

类别以 SN 前缀判定（与 multi_sensor_viewer / capture_config 中的 GF515/GF225
约定一致）：

- hand（灵巧手 / 手指）：GF515 系列，需要左右手指映射标定；
- gripper（夹爪）：GF225 系列，直接以 SN 录制。

Home 页据此渲染类别卡片，DevicePanel 据此过滤设备列表，
ViewerPage 据此只显示当前类别的传感器。
"""

SENSOR_CATEGORIES = [
    {
        "id": "hand",
        "title": "五指灵巧手传感器",
        "subtitle": "GF515 系列指灵巧手传感器",
        "prefixes": ["GF515"],
        "color": "#60a5fa",
        "badge": "手",
        "image": "hand.png",
        "features": [
            "多指触觉实时成像",
            "六维力 (Fx/Fy/Fz/Mx/My/Mz) 估计",
            "左右手指映射标定",
            "三维标记点云与深度重建",
        ],
        "button": "进入灵巧手查看器",
    },
    {
        "id": "gripper",
        "title": "两指夹爪传感器",
        "subtitle": "GF225 / GF220",
        "prefixes": ["GF225","GF220"],
        "color": "#f59e0b",
        "badge": "爪",
        "image": "225.png",
        "features": [
            "高精度触觉接触成像",
            "接触力分布可视化",
            "实时深度图与标记点云",
            "即插即用，轻量部署",
        ],
        "button": "进入夹爪查看器",
    },
]


def get_category(cat_id):
    """按 id 取类别定义，找不到返回 None。"""
    for c in SENSOR_CATEGORIES:
        if c["id"] == cat_id:
            return c
    return None


def category_for_sn(sn):
    """返回某 SN 所属类别 id；未知型号返回 None。"""
    if not sn:
        return None
    u = sn.upper()
    for c in SENSOR_CATEGORIES:
        for p in c["prefixes"]:
            if u.startswith(p.upper()):
                return c["id"]
    return None


def filter_sns_by_category(sns, cat_id):
    """保留属于给定类别的 SN（cat_id 为 None 表示不过滤）。"""
    if cat_id is None:
        return list(sns)
    return [s for s in sns if category_for_sn(s) == cat_id]
