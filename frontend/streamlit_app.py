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
DEFAULT_RETURN_PROBS = True
DEFAULT_RETURN_IMAGE = False

st.set_page_config(
    page_title="Segmentation of medical images",
    layout="wide",
)
st.title("Обработка и анализ медицинских изображений")
st.caption(
    "Загрузите изображение кожного образования и система автоматически выполнит анализ и покажет результат."
)

# Секции
tabs = st.tabs(
    [
        "Анализ изображения",
        "История обработанных изображений",
        "Статистика сервиса",
        "Проверка состояния",
    ]
)

# Анализ изображения
with tabs[0]:
    st.markdown("## Анализ изображения", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader(
            "Выберите изображение", type=["jpg", "jpeg", "png"]
        )
        top_k = st.number_input(
            "Сколько вариантов диагноза показать?",
            min_value=1,
            max_value=7,
            value=DEFAULT_TOP_K,
            help="Будет показано несколько наиболее вероятных диагнозов",
        )
        rp = DEFAULT_RETURN_PROBS
        ri = DEFAULT_RETURN_IMAGE

        send_btn = st.button("Проанализировать", key="send")
    with col2:
        preview_header = st.empty()
        preview_placeholder = st.empty()

    result_placeholder = st.empty()
    table_placeholder = st.empty()
    chart_placeholder = st.empty()

    file_bytes = None
    if uploaded:
        try:
            file_bytes = uploaded.read()
            img = Image.open(io.BytesIO(file_bytes))
            preview_placeholder.image(img, use_container_width=True)
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
                    "image": (
                        getattr(uploaded, "name", "upload.jpg"),
                        file_bytes,
                        getattr(uploaded, "type", "image/jpeg"),
                    )
                }
                headers = {
                    "x-top-k": str(top_k),
                    "x-return-probs": "true" if rp else "false",
                    "x-return-image": "true" if ri else "false",
                }
                with st.spinner("Изображение обрабатывается, подождите..."):
                    r = requests.post(
                        f"{BACKEND_URL.rstrip('/')}/forward",
                        files=files,
                        headers=headers,
                        timeout=60,
                    )
                if r.status_code != 200:
                    st.error(f"Ошибка от сервера: {r.status_code} — {r.text}")
                else:
                    data = r.json()

                    pred_code = data.get("predicted_class")
                    friendly_name = class_map.get(pred_code, pred_code)

                    elapsed = data.get("elapsed_ms")
                    fmt_elapsed = (
                        f"{elapsed:.2f} ms"
                        if isinstance(elapsed, (int, float))
                        else "N/A"
                    )

                    preview_header.markdown("**Загруженное изображение**")

                    with result_placeholder.container():
                        res_col, time_col = st.columns([3, 1])
                        res_col.markdown(f"### Результат анализа: **{friendly_name}**")
                        time_col.metric(label="Время анализа", value=f"{fmt_elapsed}")

                    tk = data.get("top_k", [])
                    if tk:
                        df_tk = pd.DataFrame(tk)
                        df_tk["Класс"] = (
                            df_tk["class"].map(class_map).fillna(df_tk["class"])
                        )
                        df_tk["Вероятность (%)"] = (
                            df_tk["prob"].astype(float) * 100
                        ).round(2)
                        df_disp = df_tk[["Класс", "Вероятность (%)"]].rename(
                            columns={
                                "Класс": "Класс",
                                "Вероятность (%)": "Вероятность (%)",
                            }
                        )
                        table_placeholder.table(df_disp)
                    else:
                        table_placeholder.empty()

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
                            title="Вероятности (все классы, %)",
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                    else:
                        chart_placeholder.empty()

                    img_b64 = data.get("image_b64")
                    if img_b64:
                        try:
                            server_img = Image.open(
                                io.BytesIO(base64.b64decode(img_b64))
                            )
                            preview_placeholder.image(
                                server_img,
                                caption="Image returned from server",
                                use_container_width=True,
                            )
                        except Exception:
                            st.info(
                                "Не удалось декодировать изображение из ответа сервера."
                            )

            except requests.exceptions.RequestException as re:
                st.error(f"Ошибка сети: {re}")
            except Exception as e:
                st.exception(e)


# История запросов
with tabs[1]:
    st.markdown("## История запросов", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        лимит = st.number_input(
            "Лимит (сколько записей загрузить)", min_value=1, max_value=1000, value=100
        )
        смещение = st.number_input("Смещение (offset)", min_value=0, value=0)
        if st.button("Загрузить историю"):
            try:
                r = requests.get(
                    f"{BACKEND_URL.rstrip('/')}/history?limit={лимит}&offset={смещение}",
                    timeout=30,
                )
                if r.status_code == 200:
                    hist = r.json()
                    if not hist:
                        st.info("История вызовов пока пустая.")
                    else:
                        df_hist = pd.DataFrame(hist)

                        df_show = df_hist.copy()

                        if "timestamp" in df_show.columns:
                            df_show["Время"] = pd.to_datetime(
                                df_show["timestamp"], errors="coerce"
                            )
                            df_show = df_show.drop(columns=["timestamp"])

                        if "image_size" in df_show.columns:
                            try:
                                sizes = pd.DataFrame(
                                    df_show["image_size"].tolist(), index=df_show.index
                                )
                                sizes.columns = ["Ширина", "Высота"][: sizes.shape[1]]
                                df_show = pd.concat(
                                    [df_show.drop(columns=["image_size"]), sizes],
                                    axis=1,
                                )
                            except Exception:
                                pass

                        rename_map = {}
                        if "elapsed_ms" in df_show.columns:
                            rename_map["elapsed_ms"] = "Время обработки (ms)"
                        if "predicted_class" in df_show.columns:
                            rename_map["predicted_class"] = "Предсказанный класс"
                        if "id" in df_show.columns:
                            rename_map["id"] = "id"

                        if rename_map:
                            df_show = df_show.rename(columns=rename_map)

                        cols_order = []
                        for c in [
                            "id",
                            "Время",
                            "Время обработки (ms)",
                            "Ширина",
                            "Высота",
                            "Предсказанный класс",
                        ]:
                            if c in df_show.columns:
                                cols_order.append(c)

                        if cols_order:
                            st.dataframe(df_show[cols_order])
                        else:
                            st.dataframe(df_show)

                else:
                    try:
                        body = r.json()
                    except Exception:
                        body = r.text
                    st.error(f"Ошибка: {r.status_code} — {body}")
            except requests.exceptions.RequestException as re:
                st.error(f"Ошибка сети: {re}")


# Статистика
with tabs[2]:
    st.markdown("## Статистика обработанных изображений", unsafe_allow_html=True)
    if st.button("Посмотреть статистику"):
        try:
            r = requests.get(f"{BACKEND_URL.rstrip('/')}/stats", timeout=30)
            if r.status_code == 200:
                js = r.json()
                time = js.get("time") or js

                st.metric("Количество образцов", time.get("count"))

                st.write("**Время (ms)**")
                # используем st.dataframe с ограничением размера
                df_time = pd.DataFrame([time]).T.rename(columns={0: "Значение"})
                st.dataframe(df_time, width=420, height=200)

                iw = js.get("image_width", {})
                ih = js.get("image_height", {})

                st.write("**Ширина изображения**")
                if iw:
                    df_iw = pd.DataFrame([iw]).T.rename(columns={0: "Значение"})
                    # уже и не слишком широкая таблица
                    st.dataframe(df_iw, width=320, height=140)
                else:
                    st.write("Нет данных по ширине")

                st.write("**Высота изображения**")
                if ih:
                    df_ih = pd.DataFrame([ih]).T.rename(columns={0: "Значение"})
                    st.dataframe(df_ih, width=320, height=140)
                else:
                    st.write("Нет данных по высоте")

            elif r.status_code == 204:
                st.info("Данных нет")
            else:
                st.error(f"Ошибка: {r.status_code} — {r.text}")
        except requests.exceptions.RequestException as re:
            st.error(f"Ошибка сети: {re}")


# Состояние сервиса
with tabs[3]:
    st.markdown("## Состояние сервиса", unsafe_allow_html=True)
    if st.button("Проверить статус"):
        try:
            r = requests.get(f"{BACKEND_URL.rstrip('/')}/health", timeout=10)
            if r.status_code == 200:
                js = r.json()
                st.success("✅ Сервис доступен")
                st.json(js)
            else:
                st.error(f"Ошибка: {r.status_code} — {r.text}")
        except requests.exceptions.RequestException as re:
            st.error(f"Ошибка сети: {re}")
