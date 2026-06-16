import io
import base64
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

class_map = {
    "akiec": "Актинический кератоз / предраковое состояние",
    "bcc": "Базальноклеточная карцинома",
    "bkl": "Доброкачественный кератоз / пятно",
    "df": "Дерматофиброма",
    "mel": "Меланома",
    "nv": "Невус (родинка)",
    "vasc": "Сосудистое образование",
}

BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_TOP_K = 3

st.set_page_config(page_title="Skin Lesion Classifier", layout="wide")
st.title("Обработка и анализ медицинских изображений")
st.caption(
    "Загрузите изображение кожного образования, система автоматически выполнит анализ и покажет результат."
)

# Дисклеймер
st.warning(
    "⚠️ **Важно:** Это исследовательский прототип. Результаты **не являются медицинским диагнозом** и не заменяют консультацию врача-дерматолога."
)

tabs = st.tabs(
    [
        "Анализ изображения",
        "История обработанных изображений",
        "Статистика сервиса",
        "Проверка состояния",
        "Информация о модели",
    ]
)

# Анализ изображения
with tabs[0]:
    st.markdown("## Анализ изображения")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

        top_k = st.number_input(
            "Сколько вариантов диагноза показать?",
            min_value=1,
            max_value=7,
            value=DEFAULT_TOP_K,
            help="Будет показано несколько наиболее вероятных диагнозов",
        )

        mode = st.selectbox(
            "Режим анализа",
            options=["global", "windows"],
            help=(
                "Global: анализирует всё изображение целиком (быстрее). Windows: разбивает изображение на фрагменты 224×224 с шагом 112 и усредняет результат (лучше для крупных или детализированных снимков)."
            ),
        )
        send_btn = st.button("Проанализировать", key="send")

    with col2:
        preview_placeholder = st.empty()

    result_placeholder = st.empty()
    table_placeholder = st.empty()
    chart_placeholder = st.empty()

    file_bytes = None
    if uploaded:
        try:
            file_bytes = uploaded.read()
            preview_placeholder.image(
                Image.open(io.BytesIO(file_bytes)), use_container_width=True
            )
        except Exception:
            preview_placeholder.text("Не удалось отобразить превью")

    if send_btn:
        if not uploaded:
            st.warning("Сначала выберите изображение.")
        else:
            try:
                if file_bytes is None:
                    file_bytes = uploaded.read()

                files = {
                    "image": (uploaded.name, file_bytes, uploaded.type or "image/jpeg")
                }
                headers = {
                    "x-top-k": str(top_k),
                    "x-return-probs": "true",
                    "x-return-image": "false",
                    "x-mode": mode,
                }

                with st.spinner("Изображение обрабатывается…"):
                    r = requests.post(
                        f"{BACKEND_URL}/forward",
                        files=files,
                        headers=headers,
                        timeout=60,
                    )

                if r.status_code != 200:
                    st.error(f"Ошибка сервера: {r.status_code} — {r.text}")
                else:
                    data = r.json()

                    pred_code = data.get("predicted_class", "—")
                    friendly = class_map.get(pred_code, pred_code)
                    confidence = data.get("confidence")
                    elapsed = data.get("elapsed_ms")
                    used_mode = data.get("mode", mode)

                    with result_placeholder.container():
                        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                        c1.markdown(f"### Результат: **{friendly}**")
                        c2.metric(
                            "Уверенность",
                            f"{confidence*100:.1f}%" if confidence else "—",
                        )
                        c3.metric("Режим", used_mode)
                        c4.metric("Время", f"{elapsed:.0f} ms" if elapsed else "—")

                    tk = data.get("top_k", [])
                    if tk:
                        df_tk = pd.DataFrame(tk)
                        df_tk["Класс"] = (
                            df_tk["class"].map(class_map).fillna(df_tk["class"])
                        )
                        df_tk["Вероятность (%)"] = (
                            df_tk["probability"].astype(float) * 100
                        ).round(2)
                        table_placeholder.table(df_tk[["Класс", "Вероятность (%)"]])

                    probs = data.get("probs")
                    if probs:
                        dfp = pd.DataFrame(
                            list(probs.items()), columns=["class", "prob"]
                        )
                        dfp["Класс"] = dfp["class"].map(class_map).fillna(dfp["class"])
                        dfp["Вероятность (%)"] = (
                            dfp["prob"].astype(float) * 100
                        ).round(2)
                        fig = px.bar(
                            dfp.sort_values("Вероятность (%)", ascending=False),
                            x="Класс",
                            y="Вероятность (%)",
                            title="Вероятности по всем 7 классам (%)",
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)

            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка сети: {e}")
            except Exception as e:
                st.exception(e)


# История запросов
with tabs[1]:
    st.markdown("## История запросов")
    lim = st.number_input(
        "Сколько записей загрузить", min_value=1, max_value=1000, value=100
    )
    off = st.number_input("Смещение (offset)", min_value=0, value=0)

    if st.button("Загрузить историю"):
        try:
            r = requests.get(
                f"{BACKEND_URL}/history?limit={lim}&offset={off}", timeout=30
            )
            if r.status_code == 200:
                hist = r.json()
                if not hist:
                    st.info("История пустая.")
                else:
                    df = pd.DataFrame(hist)
                    if "timestamp" in df.columns:
                        df["Время"] = pd.to_datetime(df["timestamp"], errors="coerce")
                        df = df.drop(columns=["timestamp"])
                    if "image_size" in df.columns:
                        try:
                            sz = pd.DataFrame(
                                df["image_size"].tolist(), columns=["Ширина", "Высота"]
                            )
                            df = pd.concat(
                                [df.drop(columns=["image_size"]), sz], axis=1
                            )
                        except Exception:
                            pass
                    df = df.rename(
                        columns={
                            "elapsed_ms": "Время (ms)",
                            "predicted_class": "Класс",
                            "confidence": "Уверенность",
                            "mode": "Режим",
                            "architecture": "Модель",
                            "stage": "Стадия",
                        }
                    )
                    st.dataframe(df)
            else:
                st.error(f"Ошибка: {r.status_code} — {r.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка сети: {e}")


# Статистика
with tabs[2]:
    st.markdown("## Статистика обработанных изображений")
    if st.button("Посмотреть статистику"):
        try:
            r = requests.get(f"{BACKEND_URL}/stats", timeout=30)
            if r.status_code == 200:
                js = r.json()
                st.metric("Всего запросов", js.get("count"))
                cols = st.columns(4)
                cols[0].metric("Среднее (ms)", f"{js.get('mean_ms', 0):.1f}")
                cols[1].metric("P50 (ms)", f"{js.get('p50_ms',  0):.1f}")
                cols[2].metric("P95 (ms)", f"{js.get('p95_ms',  0):.1f}")
                cols[3].metric("P99 (ms)", f"{js.get('p99_ms',  0):.1f}")
            elif r.status_code == 204:
                st.info("Данных нет")
            else:
                st.error(f"Ошибка: {r.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка сети: {e}")


# Состояние сервиса
with tabs[3]:
    st.markdown("## Состояние сервиса")
    if st.button("Проверить статус"):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=10)
            if r.status_code == 200:
                st.success("✅ Сервис доступен")
                st.json(r.json())
            else:
                st.error(f"Ошибка: {r.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка сети: {e}")


# Информация о модели
with tabs[4]:
    st.markdown("## Информация о модели")
    if st.button("Загрузить информацию о модели"):
        try:
            r = requests.get(f"{BACKEND_URL}/model-info", timeout=10)
            if r.status_code == 200:
                info = r.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Архитектура", info.get("architecture"))
                c2.metric("Стадия", info.get("stage"))
                c3.metric("Название", info.get("model_name"))
                st.markdown(f"**Классы:** {', '.join(info.get('classes', []))}")
                st.markdown(
                    f"**Режимы инференса:** {', '.join(info.get('inference_modes', []))}"
                )
                st.markdown("---")
                st.json(info)
            else:
                st.error(f"Ошибка: {r.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка сети: {e}")
