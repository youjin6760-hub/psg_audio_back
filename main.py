# backend/main.py

from fastapi import FastAPI, UploadFile, File, status, HTTPException, Query
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import logging
import os
import shutil
from datetime import datetime, timedelta, timezone, time  
from typing import Optional

# import librosa
import pymysql

# ============================
# TF-Hub 罹먯떆 ?붾젆?곕━ ?ㅼ젙
# ============================
TFHUB_CACHE_DIR = "/tmp/tfhub_cache_psg"

if os.path.exists(TFHUB_CACHE_DIR) and not os.path.isdir(TFHUB_CACHE_DIR):
    os.remove(TFHUB_CACHE_DIR)

os.makedirs(TFHUB_CACHE_DIR, exist_ok=True)
os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE_DIR

# ============================
# 湲곕낯 ?ㅼ젙
# ============================

KST = timezone(timedelta(hours=9))

# 濡쒓퉭 ?ㅼ젙
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI ??app = FastAPI()

# CORS ?ㅼ젙 (?꾨줎?몄뿏??
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 1. 紐⑤뜽 濡쒕뱶 (Keras + YAMNet)
# ============================

# YAMNet TF-Hub ?몃뱾
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
yamnet_model = None  # lazy loading

def get_yamnet_model():
    """
    TF-Hub?먯꽌 YAMNet 紐⑤뜽??lazy-loading?쇰줈 遺덈윭?ㅻ뒗 ?⑥닔
    """
    global yamnet_model
    if yamnet_model is None:
        logger.info("Loading YAMNet model from TF-Hub...")
        yamnet_model = hub.load(YAMNET_HANDLE)
        logger.info("YAMNet loaded.")
    return yamnet_model

# YAMNet spectrogram???낅젰?쇰줈 諛쏅뒗 Keras 紐⑤뜽 (best_model.keras)
MODEL_PATH = "models/best_model.keras"
logger.info(f"Loading Keras model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
logger.info("Keras model loaded.")

# ?대옒??留ㅽ븨 (0~4)
LABEL_MAP = {
    0: "Normal",             # ?뺤긽 ?명씉
    1: "Hypopnea",           # ??명씉
    2: "Mixed Apnea",        # ?쇳빀??臾댄샇??    3: "Obstructive Apnea",  # ?먯뇙??臾댄샇??    # 4: "Central Apnea",      # 以묒텛??臾댄샇??}

# ============================
# 鍮꾨?踰덊샇 ?댁떆 ?좏떥 (bcrypt)
# ============================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# ============================
# Pydantic ?ㅽ궎留?# ============================

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    birth_date: str   # "YYYY-MM-DD"
    gender: str       # 'Male' | 'Female' | 'Other'

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserSettingRequest(BaseModel):
    """
    ?⑤낫??3?붾㈃(?섎㈃?쒓컙, 媛곸꽦 泥댄겕 ?щ?, ?뚮엺 ?щ?)??    ??踰덉뿉 ?쒕쾭濡?蹂대궡???붿껌 諛붾뵒
    """
    user_id: int
    sleep_time: time               # ?? "23:00" -> time(23, 0)
    wake_up_time: time             # ?? "07:00" -> time(7, 0)
    is_awake_check_enabled: bool
    is_alarm_enabled: bool

# [異붽?] ?꾨줈???섏젙???ㅽ궎留?class UserProfileUpdate(BaseModel):
    """
    Settings ?붾㈃?먯꽌 ?꾨줈???섏젙 ???ъ슜?섎뒗 ?붿껌 諛붾뵒
    (?꾨줎?몃뒗 camelCase: birthDate)
    """
    name: str
    birthDate: Optional[str] = None  # "YYYY-MM-DD"
    gender: Optional[str] = None  # 'Male' | 'Female' | 'Other'


# [異붽?] ?좉?留??섏젙?????ъ슜?섎뒗 ?ㅽ궎留?class UserSettingToggleRequest(BaseModel):
    """
    Settings ?붾㈃?먯꽌 媛곸꽦 ?뚮┝ / ?뚮엺 ?좉?留?蹂寃쏀븷 ???ъ슜?섎뒗 ?붿껌 諛붾뵒
    (sleep_time, wake_up_time? 嫄대뱶由ъ? ?딆쓬)
    """
    is_awake_check_enabled: bool
    is_alarm_enabled: bool

# [異붽?] SleepSettings?먯꽌 痍⑥묠/湲곗긽 ?쒓컙留??섏젙?????ъ슜?섎뒗 ?ㅽ궎留?class SleepScheduleUpdate(BaseModel):
    sleep_time: time  # "23:00"
    wake_up_time: time  # "07:00"



# ============================
# 2. DB ?곌껐 ?ы띁
# ============================

def get_db_connection():
    """
    MySQL ?곌껐 ?⑥닔.
    MySQL ?쒕쾭??time_zone? '+09:00' (KST)濡??ㅼ젙?섏뼱 ?덈떎怨?媛??
    """
    return pymysql.connect(
        host="127.0.0.1",
        user="sleepapp",       # ?ㅼ젣 怨꾩젙?쇰줈 蹂寃?媛??        password="sleep1234",
        db="deepsleepDB",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def update_sleep_report(user_id: int, label: int):
    """
    Sleep_Report ?뚯씠釉붿뿉 ?ㅻ뒛??由ы룷??row瑜??앹꽦/?낅뜲?댄듃?쒕떎.

    - report_date: CURDATE() (MySQL, KST 湲곗?)
    - 泥?insert ??
        start_analysis_time = NOW()
        end_analysis_time   = NOW()
        apnea_level_X       = 1 (?대떦 ?쇰꺼留?1, ?섎㉧吏 0)
    - ?대? 議댁옱?쒕떎硫?
        end_analysis_time     = NOW()
        apnea_level_X         = apnea_level_X + 1
        sleep_analysis_time   = TIMEDIFF(NOW(), start_analysis_time)
    """
    if label not in [0, 1, 2, 3]:
        label = 0

    inc = [0, 0, 0, 0]
    inc[label] = 1
    inc0, inc1, inc2, inc3 = inc

    sql = """
    INSERT INTO sleep_report (
        user_id,
        report_date,
        start_analysis_time,
        end_analysis_time,
        sleep_analysis_time,
        apnea_level_0,
        apnea_level_1,
        apnea_level_2,
        apnea_level_3
    ) VALUES (
        %s,
        CURDATE(),
        NOW(),
        NOW(),
        NULL,
        %s, %s, %s, %s
    )
    ON DUPLICATE KEY UPDATE
        end_analysis_time = NOW(),
        apnea_level_0 = apnea_level_0 + VALUES(apnea_level_0),
        apnea_level_1 = apnea_level_1 + VALUES(apnea_level_1),
        apnea_level_2 = apnea_level_2 + VALUES(apnea_level_2),
        apnea_level_3 = apnea_level_3 + VALUES(apnea_level_3),
        sleep_analysis_time = TIMEDIFF(NOW(), start_analysis_time);
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, inc0, inc1, inc2, inc3))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_sleep_report(user_id: int, report_date: Optional[str] = None):
    """
    Sleep_Report?먯꽌 ?뱀젙 user + date 由ы룷????嫄댁쓣 媛?몄삩??
    report_date媛 None?대㈃ MySQL CURDATE() 湲곗? ?ㅻ뒛 ?좎쭨 由ы룷??
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if report_date is None:
                sql = """
                    SELECT *
                    FROM sleep_report
                    WHERE user_id = %s
                      AND report_date = CURDATE()
                    LIMIT 1;
                """
                cur.execute(sql, (user_id,))
            else:
                sql = """
                    SELECT *
                    FROM sleep_report
                    WHERE user_id = %s
                      AND report_date = %s
                    LIMIT 1;
                """
                cur.execute(sql, (user_id, report_date))
            row = cur.fetchone()
        return row
    finally:
        conn.close()

def upsert_user_setting(data: UserSettingRequest):
    """
    User_Setting ?뚯씠釉붿뿉 ?좎? ?ㅼ젙??????섏젙?섍퀬,
    User.has_completed_onboarding ??1濡??낅뜲?댄듃?쒕떎.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1) ?ъ슜??議댁옱 ?щ? ?뺤씤
            cur.execute("SELECT 1 FROM user WHERE user_id = %s LIMIT 1", (data.user_id,))
            exists_user = cur.fetchone()
            if not exists_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"議댁옱?섏? ?딅뒗 ?ъ슜??ID?낅땲?? {data.user_id}",
                )

            # 2) User_Setting 議댁옱 ?щ? ?뺤씤
            cur.execute(
                "SELECT setting_id FROM user_setting WHERE user_id = %s LIMIT 1",
                (data.user_id,),
            )
            row = cur.fetchone()

            # datetime.time -> "HH:MM:SS" 臾몄옄?대줈 蹂??            sleep_time_str = data.sleep_time.strftime("%H:%M:%S")
            wake_up_time_str = data.wake_up_time.strftime("%H:%M:%S")

            if not row:
                # INSERT
                insert_sql = """
                    INSERT INTO user_setting (
                        user_id,
                        sleep_time,
                        wake_up_time,
                        is_awake_check_enabled,
                        is_alarm_enabled,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, NOW()
                    )
                """
                cur.execute(
                    insert_sql,
                    (
                        data.user_id,
                        sleep_time_str,
                        wake_up_time_str,
                        int(data.is_awake_check_enabled),
                        int(data.is_alarm_enabled),
                    ),
                )
                logger.info(
                    "[upsert_user_setting] inserted user_setting for user_id=%s",
                    data.user_id,
                )
            else:
                # UPDATE
                update_sql = """
                    UPDATE user_setting
                    SET
                        sleep_time = %s,
                        wake_up_time = %s,
                        is_awake_check_enabled = %s,
                        is_alarm_enabled = %s,
                        created_at = NOW()  -- '?ㅼ젙 ?앹꽦/蹂寃??쒓컖' ?⑸룄
                    WHERE user_id = %s
                """
                cur.execute(
                    update_sql,
                    (
                        sleep_time_str,
                        wake_up_time_str,
                        int(data.is_awake_check_enabled),
                        int(data.is_alarm_enabled),
                        data.user_id,
                    ),
                )
                logger.info(
                    "[upsert_user_setting] updated user_setting for user_id=%s",
                    data.user_id,
                )

            # 3) ?⑤낫???꾨즺 ?뚮옒洹??낅뜲?댄듃
            cur.execute(
                """
                UPDATE user
                SET has_completed_onboarding = 1
                WHERE user_id = %s
                """,
                (data.user_id,),
            )
            logger.info(
                "[upsert_user_setting] set has_completed_onboarding=1 for user_id=%s",
                data.user_id,
            )

        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error("[upsert_user_setting] error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"?ъ슜???ㅼ젙 ???以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}",
        )
    finally:
        conn.close()



# ============================
# 4. ?뚯썝媛??/ 濡쒓렇??API
# ============================

@app.post("/api/signup")
async def api_signup(payload: SignupRequest):
    """
    ?뚯썝媛??
    - ?대찓??以묐났 泥댄겕
    - 鍮꾨?踰덊샇 bcrypt ?댁떆 ???    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # ?대찓??以묐났 ?뺤씤
            cur.execute("SELECT user_id FROM user WHERE email = %s", (payload.email,))
            exists = cur.fetchone()
            if exists:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="?대? 媛?낅맂 ?대찓?쇱엯?덈떎."
                )

            hashed_pw = hash_password(payload.password)

            sql = """
                INSERT INTO user (name, birth_date, gender, email, password_hash)
                VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(
                sql,
                (payload.name, payload.birth_date, payload.gender, payload.email, hashed_pw)
            )
            user_id = cur.lastrowid
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"?뚯썝媛??泥섎━ 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}"
        )
    finally:
        conn.close()

    return {
        "message": "?뚯썝媛???깃났",
        "user": {
            "user_id": user_id,
            "name": payload.name,
            "email": payload.email,
            "birth_date": payload.birth_date,
            "gender": payload.gender,
            "hasCompletedOnboarding": False,
        }
    }


@app.post("/api/login")
async def api_login(payload: LoginRequest):
    """
    濡쒓렇??
    - ?대찓?쇰줈 ?ъ슜??議고쉶
    - bcrypt 鍮꾨?踰덊샇 寃利?    - User.has_completed_onboarding 洹몃?濡??ъ슜
    """
    logger.info("[api_login] called: email=%s", payload.email)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT
                    user_id,
                    name,
                    birth_date,
                    gender,
                    email,
                    password_hash,
                    has_completed_onboarding
                FROM user
                WHERE email = %s
                LIMIT 1
            """
            cur.execute(sql, (payload.email,))
            row = cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="議댁옱?섏? ?딅뒗 ?대찓?쇱엯?덈떎."
                )

            if not verify_password(payload.password, row["password_hash"]):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="鍮꾨?踰덊샇媛 ?щ컮瑜댁? ?딆뒿?덈떎."
                )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"濡쒓렇??泥섎━ 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}"
        )
    finally:
        conn.close()

    return {
        "message": "濡쒓렇???깃났",
        "user": {
            "user_id": row["user_id"],
            "name": row["name"],
            "email": row["email"],
            "birth_date": row["birth_date"].isoformat() if row["birth_date"] else None,
            "gender": row["gender"],
            "hasCompletedOnboarding": bool(row["has_completed_onboarding"]),
        }
    }



# ============================
# 5. ?꾨줈??/ ?ㅼ젙 議고쉶쨌?섏젙 API
# ============================

# [異붽?] ?ъ슜???꾨줈??議고쉶
@app.get("/api/users/{user_id}/profile")
async def api_get_user_profile(user_id: int):
    """
    ?ъ슜???꾨줈??議고쉶:
    - User ?뚯씠釉붿뿉??name, birth_date, gender, email 議고쉶
    - ?섏씠???쒕쾭?먯꽌 怨꾩궛?댁꽌 ?대젮以?    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, name, birth_date, gender, email
                FROM user
                WHERE user_id = %s
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"議댁옱?섏? ?딅뒗 ?ъ슜??ID?낅땲?? {user_id}",
            )

        birth = row["birth_date"]
        age = None
        if birth:
            today = datetime.now(KST).date()
            age = today.year - birth.year - (
                (today.month, today.day) < (birth.month, birth.day)
            )

        return {
            "user_id": row["user_id"],
            "name": row["name"],
            "email": row["email"],
            "birthDate": birth.strftime("%Y-%m-%d") if birth else None,
            "gender": row["gender"],
            "age": age,
        }

    finally:
        conn.close()


# [異붽?] ?ъ슜???꾨줈???섏젙
@app.put("/api/users/{user_id}/profile")
async def api_update_user_profile(user_id: int, payload: UserProfileUpdate):
    """
    ?ъ슜???꾨줈???섏젙:
    - name, birth_date, gender ?낅뜲?댄듃
    - ?낅뜲?댄듃 ??理쒖떊 ?꾨줈?꾩쓣 ?ㅼ떆 諛섑솚
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # ?ъ슜??議댁옱 ?щ? ?뺤씤
            cur.execute("SELECT 1 FROM user WHERE user_id = %s LIMIT 1", (user_id,))
            exists = cur.fetchone()
            if not exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"議댁옱?섏? ?딅뒗 ?ъ슜??ID?낅땲?? {user_id}",
                )

            update_sql = """
                UPDATE user
                SET
                    name = %s,
                    birth_date = %s,
                    gender = %s
                WHERE user_id = %s
            """
            cur.execute(
                update_sql,
                (
                    payload.name,
                    payload.birthDate,  # "YYYY-MM-DD" ?먮뒗 None
                    payload.gender,
                    user_id,
                ),
            )

        conn.commit()

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error("[api_update_user_profile] error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"?꾨줈???낅뜲?댄듃 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}",
        )
    finally:
        conn.close()

    # ?낅뜲?댄듃 ??理쒖떊 媛??ㅼ떆 議고쉶?댁꽌 諛섑솚
    return await api_get_user_profile(user_id)


# [異붽?] ?ㅼ젙 議고쉶 (?섎㈃ ?쒓컙 + ?좉? ?곹깭)
@app.get("/api/users/{user_id}/settings")
async def api_get_user_settings(user_id: int):
    """
    Settings ?붾㈃?먯꽌 ?ъ슜???ъ슜???ㅼ젙 議고쉶:
    - User_Setting?먯꽌 sleep_time, wake_up_time, is_awake_check_enabled, is_alarm_enabled 議고쉶
    - ?놁쑝硫?湲곕낯媛믪쑝濡??묐떟
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sleep_time, wake_up_time, is_awake_check_enabled, is_alarm_enabled
                FROM user_setting
                WHERE user_id = %s
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()

        if not row:
            # ?꾩쭅 ?⑤낫?⑹쓣 ???덇굅???ㅼ젙???놁쓣 ??湲곕낯媛?            return {
                "user_id": user_id,
                "sleep_time": None,
                "wake_up_time": None,
                "is_awake_check_enabled": False,
                "is_alarm_enabled": False,
            }


        def fmt_time(t):
            if t is None:
                return None

            # pymysql??TIME??timedelta濡?諛섑솚?섎뒗 寃쎌슦 泥섎━
            if isinstance(t, timedelta):
                total_seconds = int(t.total_seconds())
                total_seconds = total_seconds % (24 * 3600)   # 24?쒓컙 ?⑥쐞濡?蹂댁젙
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                return f"{hours:02d}:{minutes:02d}"

            # datetime.time ??낆씤 寃쎌슦
            return t.strftime("%H:%M")

        return {
            "user_id": user_id,
            "sleep_time": fmt_time(row["sleep_time"]),
            "wake_up_time": fmt_time(row["wake_up_time"]),
            "is_awake_check_enabled": bool(row["is_awake_check_enabled"]),
            "is_alarm_enabled": bool(row["is_alarm_enabled"]),
        }

    finally:
        conn.close()



# [異붽?] ?좉?留??섏젙 (媛곸꽦 ?뚮┝ / ?뚮엺)
@app.put("/api/users/{user_id}/settings")
async def api_update_user_settings(user_id: int, payload: UserSettingToggleRequest):
    """
    Settings ?붾㈃?먯꽌 媛곸꽦 ?뚮┝ / ?뚮엺 on/off留??섏젙?????ъ슜?섎뒗 API.
    - sleep_time, wake_up_time? 嫄대뱶由ъ? ?딆쓬 (?⑤낫??/api/user-setting ?먮뒗 sleep-schedule API?먯꽌 愿由?
    - User_Setting row媛 ?놁쑝硫?湲곕낯 ?쒓컙(23:00/07:00) + ?좉? 媛믪쑝濡??앹꽦
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # User 議댁옱 ?щ? ?뺤씤
            cur.execute("SELECT 1 FROM user WHERE user_id = %s LIMIT 1", (user_id,))
            exists_user = cur.fetchone()
            if not exists_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"議댁옱?섏? ?딅뒗 ?ъ슜??ID?낅땲?? {user_id}",
                )

            # User_Setting 議댁옱 ?щ? ?뺤씤
            cur.execute(
                "SELECT sleep_time, wake_up_time FROM user_setting WHERE user_id = %s LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()

            if not row:
                # ?꾩쭅 ?ㅼ젙???녿떎硫?湲곕낯 ?쒓컙?쇰줈 INSERT
                insert_sql = """
                    INSERT INTO user_setting (
                        user_id,
                        sleep_time,
                        wake_up_time,
                        is_awake_check_enabled,
                        is_alarm_enabled,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, NOW()
                    )
                """
                cur.execute(
                    insert_sql,
                    (
                        user_id,
                        "23:00:00",
                        "07:00:00",
                        int(payload.is_awake_check_enabled),
                        int(payload.is_alarm_enabled),
                    ),
                )
            else:
                # 湲곗〈 row媛 ?덉쑝硫??좉?留??낅뜲?댄듃
                update_sql = """
                    UPDATE user_setting
                    SET
                        is_awake_check_enabled = %s,
                        is_alarm_enabled = %s
                    WHERE user_id = %s
                """
                cur.execute(
                    update_sql,
                    (
                        int(payload.is_awake_check_enabled),
                        int(payload.is_alarm_enabled),
                        user_id,
                    ),
                )

        conn.commit()

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error("[api_update_user_settings] error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"?ъ슜???ㅼ젙 ?낅뜲?댄듃 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}",
        )
    finally:
        conn.close()

    # ?낅뜲?댄듃 ??理쒖떊 ?ㅼ젙 諛섑솚
    return await api_get_user_settings(user_id)


# [異붽?] 痍⑥묠/湲곗긽 ?쒓컙留??섏젙 (SleepSettings?먯꽌 ?ъ슜)
@app.put("/api/users/{user_id}/sleep-schedule")
async def api_update_sleep_schedule(user_id: int, payload: SleepScheduleUpdate):
    """
    SleepSettings ?붾㈃?먯꽌 '痍⑥묠 ?쒓컙 / 湲곗긽 ?쒓컙'???섏젙?????ъ슜?섎뒗 API.

    - is_awake_check_enabled, is_alarm_enabled 媛믪? 湲곗〈 User_Setting 媛??좎?
      (?놁쑝硫?湲곕낯媛믪쑝濡?INSERT)
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1) User 議댁옱 ?щ? ?뺤씤
            cur.execute("SELECT 1 FROM user WHERE user_id = %s LIMIT 1", (user_id,))
            exists_user = cur.fetchone()
            if not exists_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"議댁옱?섏? ?딅뒗 ?ъ슜??ID?낅땲?? {user_id}",
                )

            # 2) 湲곗〈 ?ㅼ젙 議고쉶
            cur.execute(
                """
                SELECT sleep_time, wake_up_time, is_awake_check_enabled, is_alarm_enabled
                FROM user_setting
                WHERE user_id = %s
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()

            sleep_time_str = payload.sleep_time.strftime("%H:%M:%S")
            wake_up_time_str = payload.wake_up_time.strftime("%H:%M:%S")

            if not row:
                # ?꾩쭅 ?ㅼ젙???놁쑝硫?湲곕낯 ?좉?媛??꾧린)?쇰줈 INSERT
                insert_sql = """
                    INSERT INTO user_setting (
                        user_id,
                        sleep_time,
                        wake_up_time,
                        is_awake_check_enabled,
                        is_alarm_enabled,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, NOW()
                    )
                """
                cur.execute(
                    insert_sql,
                    (
                        user_id,
                        sleep_time_str,
                        wake_up_time_str,
                        0,  # is_awake_check_enabled 湲곕낯 0
                        0,  # is_alarm_enabled 湲곕낯 0
                    ),
                )
                logger.info(
                    "[api_update_sleep_schedule] inserted user_setting for user_id=%s",
                    user_id,
                )
            else:
                # 湲곗〈 row媛 ?덉쑝硫?痍⑥묠/湲곗긽 ?쒓컙留??낅뜲?댄듃
                update_sql = """
                    UPDATE user_setting
                    SET
                        sleep_time = %s,
                        wake_up_time = %s
                    WHERE user_id = %s
                """
                cur.execute(
                    update_sql,
                    (
                        sleep_time_str,
                        wake_up_time_str,
                        user_id,
                    ),
                )
                logger.info(
                    "[api_update_sleep_schedule] updated sleep_time/wake_up_time for user_id=%s",
                    user_id,
                )

        conn.commit()

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error("[api_update_sleep_schedule] error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"?섎㈃ ?⑦꽩 ?낅뜲?댄듃 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}",
        )
    finally:
        conn.close()

    # ?낅뜲?댄듃 ??理쒖떊 ?ㅼ젙 諛섑솚
    return await api_get_user_settings(user_id)



# @app.put("/api/users/{user_id}/settings")
# async def api_update_user_settings(user_id: int, payload: UserSettingToggleRequest):
#     """
#     Settings ?붾㈃?먯꽌 媛곸꽦 ?뚮┝ / ?뚮엺 on/off留??섏젙?????ъ슜?섎뒗 API.
#     - sleep_time, wake_up_time? 嫄대뱶由ъ? ?딆쓬 (?⑤낫??/api/user-setting?먯꽌 愿由?
#     - User_Setting row媛 ?놁쑝硫?湲곕낯 ?쒓컙(23:00/07:00) + ?좉? 媛믪쑝濡??앹꽦
#     """
#     conn = get_db_connection()
#     try:
#         with conn.cursor() as cur:
#             # User 議댁옱 ?щ? ?뺤씤
#             cur.execute("SELECT 1 FROM User WHERE user_id = %s LIMIT 1", (user_id,))
#             exists_user = cur.fetchone()
#             if not exists_user:
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"議댁옱?섏? ?딅뒗 ?ъ슜??ID?낅땲?? {user_id}",
#                 )

#             # User_Setting 議댁옱 ?щ? ?뺤씤
#             cur.execute(
#                 "SELECT sleep_time, wake_up_time FROM User_Setting WHERE user_id = %s LIMIT 1",
#                 (user_id,),
#             )
#             row = cur.fetchone()

#             if not row:
#                 # ?꾩쭅 ?ㅼ젙???녿떎硫?湲곕낯 ?쒓컙?쇰줈 INSERT
#                 insert_sql = """
#                     INSERT INTO User_Setting (
#                         user_id,
#                         sleep_time,
#                         wake_up_time,
#                         is_awake_check_enabled,
#                         is_alarm_enabled,
#                         created_at
#                     ) VALUES (
#                         %s, %s, %s, %s, %s, NOW()
#                     )
#                 """
#                 cur.execute(
#                     insert_sql,
#                     (
#                         user_id,
#                         "23:00:00",
#                         "07:00:00",
#                         int(payload.is_awake_check_enabled),
#                         int(payload.is_alarm_enabled),
#                     ),
#                 )
#             else:
#                 # 湲곗〈 row媛 ?덉쑝硫??좉?留??낅뜲?댄듃
#                 update_sql = """
#                     UPDATE User_Setting
#                     SET
#                         is_awake_check_enabled = %s,
#                         is_alarm_enabled = %s
#                     WHERE user_id = %s
#                 """
#                 cur.execute(
#                     update_sql,
#                     (
#                         int(payload.is_awake_check_enabled),
#                         int(payload.is_alarm_enabled),
#                         user_id,
#                     ),
#                 )

#         conn.commit()

#     except HTTPException:
#         conn.rollback()
#         raise
#     except Exception as e:
#         conn.rollback()
#         logger.error("[api_update_user_settings] error: %s", e)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"?ъ슜???ㅼ젙 ?낅뜲?댄듃 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}",
#         )
#     finally:
#         conn.close()

#     # ?낅뜲?댄듃 ??理쒖떊 ?ㅼ젙 諛섑솚
#     return await api_get_user_settings(user_id)







# ============================
# 6. ?ъ슜???ㅼ젙 API
# ============================

@app.post("/api/user-setting")
async def api_save_user_setting(payload: UserSettingRequest):
    logger.info(
        "[api_save_user_setting] called: user_id=%s, sleep_time=%s, wake_up_time=%s, awake_check=%s, alarm=%s",
        payload.user_id,
        payload.sleep_time,
        payload.wake_up_time,
        payload.is_awake_check_enabled,
        payload.is_alarm_enabled,
    )

    upsert_user_setting(payload)

    return {
        "status": "ok",
        "message": "?ъ슜???ㅼ젙????λ릺?덉뒿?덈떎.",
        "user_id": payload.user_id,
    }

# ============================
# 5. ?ъ뒪 泥댄겕
# ============================

@app.get("/health")
async def health():
    return {"status": "ok"}

# ============================
# 6. ?뱀쓬 ?뚯씪 ????대뜑
# ============================

os.makedirs("recordings", exist_ok=True)

# # ============================
# # 7. .wav ??YAMNet ?ㅽ럺?몃줈洹몃옩 ?⑥닔
# # ============================

# def wav_to_yamnet_spectrogram(path: str, target_sr: int = 16000) -> np.ndarray:
#     """
#     .wav ?뚯씪 寃쎈줈瑜?諛쏆븘??
#       - 16kHz mono waveform 濡쒕뵫 (librosa媛 ?먮룞 由ъ깦?뚮쭅)
#       - YAMNet?쇰줈 spectrogram 異붿텧
#     """
#     logger.info("[yamnet] 1. librosa.load start (wav): %s", path)
#     y, sr = librosa.load(path, sr=target_sr, mono=True)
#     logger.info("[yamnet] 2. librosa.load done: len(y)=%d, sr=%d", len(y), sr)

#     waveform = tf.convert_to_tensor(y, dtype=tf.float32)
#     logger.info("[yamnet] 3. waveform tensor created: shape=%s", waveform.shape)

#     yamnet = get_yamnet_model()
#     logger.info("[yamnet] 4. YAMNet model loaded")

#     scores, embeddings, spectrogram = yamnet(waveform)  # spectrogram: (T, 64)
#     logger.info(
#         "[yamnet] 5. yamnet forward done: spectrogram shape=%s",
#         spectrogram.shape,
#     )

#     spec_np = spectrogram.numpy().astype("float32")
#     logger.info("[yamnet] 6. spectrogram numpy converted: shape=%s", spec_np.shape)

#     return spec_np

# ============================
# [?섏젙] WAV ??YAMNet spectrogram (tf.signal.resample ?쒓굅 踰꾩쟾)
# ============================
def wav_to_yamnet_spectrogram(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    .wav ?뚯씪??TensorFlow濡??쎌뼱???    - tf.audio.decode_wav 濡??뚯떛
    - ?꾩슂?섎㈃ numpy濡?16kHz 由ъ깦??    - YAMNet???ｌ뼱??spectrogram (3踰덉㎏ 異쒕젰)??numpy濡?諛섑솚

    諛섑솚媛? spectrogram.numpy()  # shape: (T, 64) ??    """
    logger.info("[yamnet] 1. tf.io.read_file start (wav): %s", path)
    audio_binary = tf.io.read_file(path)

    # decode_wav: [-1.0, 1.0] float32, (samples, channels)
    waveform, sample_rate = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1,   # mono 媛뺤젣
    )  # waveform: (N, 1)
    waveform = tf.squeeze(waveform, axis=-1)  # (N,)
    sample_rate = int(sample_rate.numpy())
    logger.info(
        "[yamnet] 2. tf.audio.decode_wav done: len=%d, sr=%d",
        waveform.shape[0],
        sample_rate,
    )

    # === [?듭떖 ?섏젙] numpy濡?16kHz 由ъ깦??===
    if sample_rate != target_sr:
        logger.info(
            "[yamnet] 3. numpy resample from %d Hz to %d Hz",
            sample_rate,
            target_sr,
        )
        # TF tensor -> numpy
        wave_np = waveform.numpy().astype("float32")

        # ?꾩껜 湲몄씠(珥?
        duration = wave_np.shape[0] / float(sample_rate)
        # 由ъ깦????湲몄씠
        new_length = int(round(duration * target_sr))

        # ?먮옒/???쒓컙異??앹꽦
        old_times = np.linspace(0.0, duration, num=wave_np.shape[0], endpoint=False)
        new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)

        # ?좏삎 蹂닿컙?쇰줈 由ъ깦??        wave_np = np.interp(new_times, old_times, wave_np).astype("float32")

        # ?ㅼ떆 TF tensor濡?蹂??        waveform = tf.convert_to_tensor(wave_np, dtype=tf.float32)
        logger.info(
            "[yamnet] 4. resample done: new_len=%d",
            new_length,
        )
    else:
        # ?섑뵆?덉씠?멸? ?대? 16kHz??寃쎌슦
        waveform = tf.cast(waveform, tf.float32)

    # YAMNet 紐⑤뜽 異붾줎
    yamnet = get_yamnet_model()
    logger.info("[yamnet] 5. YAMNet forward start")
    scores, embeddings, spectrogram = yamnet(waveform)
    logger.info(
        "[yamnet] 6. YAMNet forward done: spectrogram shape=%s",
        spectrogram.shape,
    )

    # best_model.keras ???섍린湲??꾪빐 numpy濡?蹂??    spec_np = spectrogram.numpy().astype("float32")
    return spec_np


# ============================
# 8. ?낅줈?쒕맂 10珥?泥?겕 泥섎━
# ============================

@app.post("/upload-audio")
async def upload_audio(
    user_id: int = Query(..., description="User ID (荑쇰━ ?뚮씪誘명꽣 ?user_id=1 ?뺥깭)"),
    file: UploadFile = File(...),
):
    """
    10珥??ㅻ뵒??泥?겕 ?낅줈????YAMNet ?꾨쿋??1024) ??test.keras ?덉륫 ??Sleep_Report ?낅뜲?댄듃.

    - user_id: 荑쇰━ ?뚮씪誘명꽣 (?user_id=1)
    - file   : Form-data濡??ㅼ뼱?ㅻ뒗 ?ㅻ뵒???뚯씪(.webm)
    """
    logger.info("[upload-audio] called (user_id=%s, filename=%s)", user_id, file.filename)

    # 0) user_id ?좏슚??泥댄겕
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            logger.info("[upload-audio] 1. checking user...")
            cur.execute("SELECT 1 FROM user WHERE user_id = %s LIMIT 1", (user_id,))
            exists = cur.fetchone()
            if not exists:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"議댁옱?섏? ?딅뒗 ?ъ슜??ID?낅땲?? {user_id}",
                )
    finally:
        conn.close()

    # 1) ?뚯씪 ???(KST 湲곗? ?뚯씪紐?
    logger.info("[upload-audio] 2. saving file...")
    filename = datetime.now(KST).strftime("%Y%m%d_%H%M%S_%f") + ".wav"
    path = os.path.join("recordings", filename)

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = -1
        logger.info(
            "[upload-audio] file saved: %s (size=%d bytes, user_id=%s)",
            path,
            file_size,
            user_id,
        )
    except Exception as e:
        logger.error("[upload-audio] failed to save file: %s (%s)", path, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"?뱀쓬 ?뚯씪 ??μ뿉 ?ㅽ뙣?덉뒿?덈떎: {e}",
        )

    # # 2) .webm ??YAMNet ?꾨쿋??1024)
    # try:
    #     logger.info("[upload-audio] 3. webm_to_yamnet_embedding start")
    #     embedding_1024 = webm_to_yamnet_embedding(path)  # (1024,)
    #     logger.info("[upload-audio] 4. webm_to_yamnet_embedding done")
    # except Exception as e:
    #     logger.error("[upload-audio] error in webm_to_yamnet_embedding: %s", e)
    #     # 蹂???ㅽ뙣 ???뚯씪 ??젣
    #     try:
    #         os.remove(path)
    #     except OSError:
    #         pass
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail=f"?ㅻ뵒??泥섎━ 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}",
    #     )


    # 2) .webm ??YAMNet ?ㅽ럺?몃줈洹몃옩?쇰줈 蹂??    try:
        logger.info("[upload-audio] 3. wav_to_yamnet_spectrogram start")
        spectrogram = wav_to_yamnet_spectrogram(path)  # (T, 64)
        logger.info(
            "[upload-audio] 4. wav_to_yamnet_spectrogram done, shape=%s",
            spectrogram.shape,
        )
    except Exception as e:
        logger.error("[upload-audio] error in wav_to_yamnet_spectrogram: %s", e)
        # 蹂???ㅽ뙣 ???뚯씪 ??젣
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"?ㅻ뵒??泥섎━ 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}",
        )    

    # 3) 紐⑤뜽 ?덉륫 (?낅젰: (1, 1024) 媛??
    try:
        logger.info("[upload-audio] 5. model.predict start")
        # x = np.expand_dims(embedding_1024, axis=0)  # (1, 1024)

        # [?섏젙] ?쒓컙異?T)???됯퇏 ?댁꽌 湲몄씠 64 踰≫꽣濡?蹂??        spec_mean = np.mean(spectrogram, axis=0).astype("float32")  # (64,)
        # [?섏젙] 紐⑤뜽??湲곕??섎뒗 (1, 64, 1) ?뺥깭濡?reshape
        x = spec_mean.reshape(1, 64, 1)

        preds = model.predict(x)
        logger.info("[upload-audio] 6. model.predict done")
        probs = preds[0]
        label = int(np.argmax(probs))
        max_prob = float(np.max(probs))
        label_name = LABEL_MAP.get(label, "Unknown")
        logger.info(
            "[upload-audio] prediction -> label=%s (%s), max_prob=%.4f",
            label,
            label_name,
            max_prob,
        )
    except Exception as e:
        logger.error("[upload-audio] model.predict error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"紐⑤뜽 ?덉륫 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}",
        )

    # 4) Sleep_Report ?낅뜲?댄듃
    try:
        logger.info("[upload-audio] 7. update_sleep_report start")
        update_sleep_report(user_id=user_id, label=label)
        logger.info("[upload-audio] 8. update_sleep_report done")
    except Exception as e:
        logger.error("[upload-audio] update_sleep_report error: %s", e)
        # DB ?먮윭 ???뚯씪? ?④꺼?먭퀬 ?먮윭 諛섑솚
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"?섎㈃ 由ы룷??媛깆떊 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {e}",
        )

    # 5) ?뺤긽(0)??寃쎌슦 ?뚯씪 ??젣, ?댁긽(1~4)? ?④꺼??異뷀썑 遺꾩꽍/?ы븰?듭슜)
    saved = True
    if label == 0:
        try:
            os.remove(path)
        except OSError:
            pass
        saved = False

    logger.info("[upload-audio] 9. returning response")

    return {
        "status": "ok",
        "user_id": user_id,
        "filename": filename,
        "label": label,
        "label_name": label_name,
        "max_prob": max_prob,
        "saved": saved,
    }

# ============================
# 9. /predict : 由ы룷??議고쉶
# ============================

@app.get("/predict")
async def predict(user_id: int, date: Optional[str] = None):
    """
    ?ㅻ뒛(?먮뒗 ?뱀젙 ?좎쭨)??Sleep_Report 由ы룷?몃? 議고쉶.

    - user_id: ?꾩닔 (荑쇰━ ?뚮씪誘명꽣)
    - date: ?좏깮 (YYYY-MM-DD 臾몄옄??
      - ?놁쑝硫?MySQL CURDATE() 湲곗? ?ㅻ뒛 ?좎쭨 由ы룷??議고쉶
    """
    try:
        report = get_sleep_report(user_id=user_id, report_date=date)
    except Exception as e:
        logger.error("[predict] Failed to fetch report: %s", e)
        return {"status": "error", "message": f"Failed to fetch report: {e}"}

    if not report:
        return {
            "status": "not_found",
            "message": "由ы룷?멸? ?놁뒿?덈떎. (?대떦 ?좎쭨???섎㈃ 遺꾩꽍 ?곗씠?곌? ?놁쓣 ???덉쓬)",
        }

    return {
        "status": "ok",
        "user_id": report["user_id"],
        "report_date": str(report["report_date"]) if report["report_date"] else None,
        "start_analysis_time": (
            report["start_analysis_time"].isoformat()
            if report["start_analysis_time"] else None
        ),
        "end_analysis_time": (
            report["end_analysis_time"].isoformat()
            if report["end_analysis_time"] else None
        ),
        "sleep_analysis_time": (
            str(report["sleep_analysis_time"])
            if report["sleep_analysis_time"] else None
        ),
        "apnea_level_0": report["apnea_level_0"],
        "apnea_level_1": report["apnea_level_1"],
        "apnea_level_2": report["apnea_level_2"],
        "apnea_level_3": report["apnea_level_3"]
    }
