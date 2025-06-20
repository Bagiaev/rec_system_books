import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from recommenderSVD import SVDRecommender
from recommenderTFIDF import ContentRecommender
from recommendedHybrid import HybridRecommender

# DB Connection
DB_URI = "postgresql://postgres:Ars8095835@localhost:5433/bookdb"
engine = create_engine(DB_URI)

# Load data
@st.cache_data
def load_books():
    return pd.read_sql("SELECT * FROM books;", engine)

@st.cache_data
def load_users():
    return pd.read_sql("SELECT * FROM users;", engine)

@st.cache_data
def load_ratings():
    return pd.read_sql("SELECT * FROM ratings;", engine)

books_df = load_books()
users_df = load_users()
ratings_df = load_ratings()

# Models
cf_model = SVDRecommender()
cb_model = ContentRecommender()
hybrid_model = HybridRecommender(books_df=books_df)

# App layout
st.set_page_config(page_title="Book Recommender", layout="wide")
st.title("📚 Book Recommendation System")

# Session state
if "page" not in st.session_state:
    st.session_state.page = "Каталог"
if "user_id" not in st.session_state:
    st.session_state.user_id = 99 # users_df.user_id.sample(1).iloc[0]  # случайный пользователь
if "selected_book_isbn" not in st.session_state:
    st.session_state.selected_book_isbn = None
# st.sidebar.write(f"📍 Current page: {st.session_state.page}")

# Sidebar Navigation
pages = ["Каталог", "Профиль", "Рекомендации"]
if st.session_state.page in pages:
    selected_page = st.sidebar.radio("Перейти на страницу:", pages, index=pages.index(st.session_state.page))
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

# Отображение текущей страницы
st.sidebar.write(f"📍 Current page: {st.session_state.page}")

# Catalog Page
if st.session_state.page == "Каталог":
    st.subheader("📖 Каталог книг")
    search = st.text_input("Поиск по названию или автору:")
    
    df = books_df.copy()
    if search:
        df = df[df.title.str.contains(search, case=False, na=False) | 
               df.author.str.contains(search, case=False, na=False)]
    
    for _, book in df.head(20).iterrows():
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(book.image_l if pd.notna(book.image_l) else "images/null_book.jpg", 
                         use_container_width=True)
            with cols[1]:
                st.subheader(book.title)
                st.write(f"**Автор:** {book.author}")
                st.write(f"**Год:** {book.year if pd.notna(book.year) else 'Не указан'}")
                st.write(f"**Издатель:** {book.publisher if pd.notna(book.publisher) else 'Не указан'}")
                button_key = f"detail_{book.isbn}_{book.title}"
                if st.button("Подробнее", key=button_key):
                    st.session_state.selected_book_isbn = book.isbn
                    st.session_state.page = "Detail"
                    st.rerun()

# Detail Page
elif st.session_state.page == "Detail" and st.session_state.selected_book_isbn:
    isbn = st.session_state.selected_book_isbn
    book = books_df[books_df.isbn == isbn].iloc[0]

    # Кнопка "Назад" вверху страницы
    if st.button("← Назад в каталог"):
        st.session_state.page = "Каталог"
        st.rerun()
    
    cols = st.columns([1, 2])
    with cols[0]:
        st.image(book.image_l if pd.notna(book.image_l) else "images/null_book.jpg", 
                 width=300)
    with cols[1]:
        st.header(book.title)
        st.write(f"**Автор:** {book.author}")
        st.write(f"**Год издания:** {book.year if pd.notna(book.year) else 'Не указан'}")
        st.write(f"**Издатель:** {book.publisher if pd.notna(book.publisher) else 'Не указан'}")
    
    # Оценка книги
    st.subheader("⭐ Оцените книгу")
    rating = st.slider("Ваша оценка (1-10)", 1, 10, 5)
    if st.button("Сохранить оценку"):
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO ratings(user_id, isbn, rating)
                VALUES(:u, :b, :r)
                ON CONFLICT (user_id, isbn)
                DO UPDATE SET rating = :r;
            """), {"u": int(st.session_state.user_id), "b": str(isbn), "r": int(rating)})
            conn.commit()
        st.success("Ваша оценка сохранена!")
        st.rerun()
    
    # Все рекомендации в табах
    tab1, tab2, tab3 = st.tabs(["Контентные рекомендации", "Коллаборативная фильтрация", "Гибридные рекомендации"])
    
    with tab1:
        st.subheader("📚 Похожие книги (контентные рекомендации)")
        try:
            cb_recs = cb_model.get_recommendations(book.title)
            if isinstance(cb_recs, pd.DataFrame) and not cb_recs.empty:
                for _, rec in cb_recs.head(5).iterrows():
                    rec_book = books_df[books_df.title == rec['title']]
                    if not rec_book.empty:
                        rec_b = rec_book.iloc[0]
                        cols = st.columns([1, 4])
                        with cols[0]:
                            st.image(rec_b.image_m if pd.notna(rec_b.image_m) else "https://via.placeholder.com/100", 
                                     width=100)
                        with cols[1]:
                            st.write(f"**{rec_b.title}**")
                            st.write(f"Автор: {rec_b.author}")
                            st.write(f"Год: {rec_b.year if pd.notna(rec_b.year) else 'Не указан'}")
            else:
                st.warning("Не удалось получить контентные рекомендации")
        except Exception as e:
            st.error(f"Ошибка при получении контентных рекомендаций: {e}")
    
    with tab2:
        st.subheader("👥 Рекомендации для вас (коллаборативная фильтрация)")
        try:
            cf_recs = cf_model.recommend_books(st.session_state.user_id)
            if not cf_recs.empty:
                for _, row in cf_recs.head(5).iterrows():
                    match = books_df[books_df.isbn == row['isbn']] if 'isbn' in row else books_df[books_df.title == row['title']]
                    if not match.empty:
                        b = match.iloc[0]
                        cols = st.columns([1, 4])
                        with cols[0]:
                            st.image(b.image_m if pd.notna(b.image_m) else "https://via.placeholder.com/100", 
                                     width=100)
                        with cols[1]:
                            st.write(f"**{b.title}**")
                            st.write(f"Автор: {b.author}")
                            st.write(f"Год: {b.year if pd.notna(b.year) else 'Не указан'}")
            else:
                st.warning("Не удалось получить рекомендации коллаборативной фильтрации")
        except Exception as e:
            st.error(f"Ошибка при получении рекомендаций CF: {e}")
    
    with tab3:
        st.subheader("✨ Гибридные рекомендации")
        try:
            hybrid_recs = hybrid_model.recommend(st.session_state.user_id, book.title)
            if not hybrid_recs.empty:
                for _, row in hybrid_recs.head(5).iterrows():
                    match = books_df[books_df.title == row['title']]
                    if not match.empty:
                        b = match.iloc[0]
                        cols = st.columns([1, 4])
                        with cols[0]:
                            st.image(b.image_m if pd.notna(b.image_m) else "https://via.placeholder.com/", 
                                    )
                        with cols[1]:
                            st.write(f"**{b.title}**")
                            st.write(f"Автор: {b.author}")
                            st.write(f"Год: {b.year if pd.notna(b.year) else 'Не указан'}")
            else:
                st.warning("Не удалось получить гибридные рекомендации")
        except Exception as e:
            st.error(f"Ошибка при получении гибридных рекомендаций: {e}")

# Profile Page
elif st.session_state.page == "Профиль":
    st.subheader("👤 Профиль пользователя")
    user = users_df[users_df.user_id == st.session_state.user_id].iloc[0]
    st.write(f"**ID пользователя:** {user.user_id}")
    st.write(f"**Местоположение:** {user.location if pd.notna(user.location) else 'Не указано'}")
    st.write(f"**Возраст:** {user.age if pd.notna(user.age) else 'Не указан'}")

    st.subheader("📚 Ваши оценки")
    user_ratings = ratings_df[ratings_df.user_id == user.user_id]
    if not user_ratings.empty:
        merged = user_ratings.merge(books_df, on="isbn")
        for _, row in merged.iterrows():
            with st.expander(f"{row['title']} - {row['rating']}/10"):
                cols = st.columns([1, 4])
                with cols[0]:
                    st.image(row['image_m'] if pd.notna(row['image_m']) else "https://via.placeholder.com/100", 
                             width=100)
                with cols[1]:
                    st.write(f"**Автор:** {row['author']}")
                    st.write(f"**Год:** {row['year'] if pd.notna(row['year']) else 'Не указан'}")
    else:
        st.info("Вы еще не оценили ни одной книги")

# Recommendations Page
elif st.session_state.page == "Рекомендации":
    st.subheader("🔮 Персональные рекомендации")
    
    tab1, tab2 = st.tabs(["Коллаборативная фильтрация", "Гибридные рекомендации"])
    
    with tab1:
        st.subheader("👥 На основе ваших предпочтений")
        try:
            cf_recs = cf_model.recommend_books(st.session_state.user_id)
            if not cf_recs.empty:
                for _, row in cf_recs.head(10).iterrows():
                    match = books_df[books_df.isbn == row['isbn']] if 'isbn' in row else books_df[books_df.title == row['title']]
                    if not match.empty:
                        b = match.iloc[0]
                        cols = st.columns([1, 4])
                        with cols[0]:
                            st.image(b.image_m if pd.notna(b.image_m) else "https://via.placeholder.com/100", 
                                     width=100)
                        with cols[1]:
                            st.write(f"**{b.title}**")
                            st.write(f"Автор: {b.author}")
                            st.write(f"Год: {b.year if pd.notna(b.year) else 'Не указан'}")
                            if st.button("Подробнее", key=f"rec_cf_{b.isbn}"):
                                st.session_state.selected_book_isbn = b.isbn
                                st.session_state.page = "Detail"
                                st.rerun()
            else:
                st.warning("Не удалось получить рекомендации")
        except Exception as e:
            st.error(f"Ошибка при получении рекомендаций: {e}")
    
    with tab2:
        st.subheader("✨ Гибридные рекомендации")
        try:
            # Выбираем случайную книгу из оцененных пользователем для примера
            rated_books = ratings_df[ratings_df.user_id == st.session_state.user_id]
            if not rated_books.empty:
                sample_book = rated_books.sample(1).iloc[0]
                book_title = books_df[books_df.isbn == sample_book.isbn].iloc[0].title
                hybrid_recs = hybrid_model.recommend(st.session_state.user_id, book_title)
                
                if not hybrid_recs.empty:
                    for _, row in hybrid_recs.head(10).iterrows():
                        match = books_df[books_df.title == row['title']]
                        if not match.empty:
                            b = match.iloc[0]
                            cols = st.columns([1, 4])
                            with cols[0]:
                                st.image(b.image_m if pd.notna(b.image_m) else "https://via.placeholder.com/100", 
                                         width=100)
                            with cols[1]:
                                st.write(f"**{b.title}**")
                                st.write(f"Автор: {b.author}")
                                st.write(f"Год: {b.year if pd.notna(b.year) else 'Не указан'}")
                                if st.button("Подробнее", key=f"rec_hybrid_{b.isbn}"):
                                    st.session_state.selected_book_isbn = b.isbn
                                    st.session_state.page = "Detail"
                                    st.rerun()
                else:
                    st.warning("Не удалось получить гибридные рекомендации")
            else:
                st.info("Оцените несколько книг, чтобы получить гибридные рекомендации")
        except Exception as e:
            st.error(f"Ошибка при получении гибридных рекомендаций: {e}")