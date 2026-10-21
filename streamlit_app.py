"""
==========================================================================
 DASHBOARD STREAMLIT - BEDAH KASUS 1, 2, 3
 Tipe Data & Visualisasi Data yang Sesuai
--------------------------------------------------------------------------
 Nama : Stefhanus Christopher Hasiholan Simarmata
 NIM  : 241051014
==========================================================================
 Cara menjalankan:
     pip install -r requirements.txt
     streamlit run streamlit_app.py
==========================================================================
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAMA = "Stefhanus Christopher Hasiholan Simarmata"
NIM = "241051014"

st.set_page_config(
    page_title=f"Bedah Kasus 1-3 | {NAMA} - {NIM}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
#  GENERATOR DATA SINTETIS (sama persis dengan yang ada di notebook)
# ======================================================================
import numpy as np
import pandas as pd

SEED = 241051019  # seed = NIM, supaya data selalu reproducible


# ----------------------------------------------------------------------
# BEDAH KASUS 1 - Dataset Operasional Penjualan
# ----------------------------------------------------------------------
KATALOG_PRODUK = {
    # produk: (kategori, harga_dasar, rata2_unit_per_transaksi, bobot_populer)
    "Laptop":            ("Elektronik",    8_500_000, 40, 0.8),
    "Smartphone":        ("Elektronik",    4_200_000, 55, 1.4),
    "Headphone":         ("Elektronik",      650_000, 70, 1.0),
    "Smartwatch":        ("Elektronik",    1_800_000, 35, 0.6),
    "Buku":              ("Pendidikan",       75_000, 120, 1.6),
    "Alat Tulis":        ("Pendidikan",       25_000, 150, 1.3),
    "Kalkulator":        ("Pendidikan",      185_000, 60, 0.7),
    "Tas Ransel":        ("Pendidikan",      280_000, 50, 0.9),
    "Kemeja":            ("Fashion",         225_000, 65, 1.2),
    "Sepatu Sneakers":   ("Fashion",         550_000, 45, 1.1),
    "Jaket":             ("Fashion",         375_000, 40, 0.8),
    "Rice Cooker":       ("Rumah Tangga",    450_000, 38, 0.8),
    "Blender":           ("Rumah Tangga",    320_000, 42, 0.7),
    "Set Peralatan Masak": ("Rumah Tangga",  600_000, 30, 0.5),
    "Matras Yoga":       ("Olahraga",        190_000, 48, 0.6),
    "Dumbbell":          ("Olahraga",        260_000, 36, 0.5),
    "Sepeda Lipat":      ("Olahraga",      4_800_000, 12, 0.3),
}


def _faktor_musiman(kategori, bulan):
    """Pola musiman sederhana per kategori (mis. Harbolnas, tahun ajaran baru)."""
    if kategori == "Elektronik" and bulan in (11, 12):
        return 1.35
    if kategori == "Pendidikan" and bulan in (1, 7, 8):
        return 1.40
    if kategori == "Fashion" and bulan in (3, 4, 12):
        return 1.30
    if kategori == "Olahraga" and bulan in (1, 9):
        return 1.20
    return 1.0


def buat_data_penjualan(n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    produk = list(KATALOG_PRODUK)
    bobot = np.array([KATALOG_PRODUK[p][3] for p in produk])
    bobot = bobot / bobot.sum()

    # Tanggal: 2 tahun, transaksi makin ramai seiring waktu (tren naik)
    hari = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    p_hari = 1 + 0.6 * np.arange(len(hari)) / len(hari)
    p_hari = p_hari / p_hari.sum()
    tanggal = np.sort(rng.choice(hari, size=n, p=p_hari))

    pilih = rng.choice(produk, size=n, p=bobot)
    rows = []
    for p, t in zip(pilih, pd.to_datetime(tanggal)):
        kat, harga_dasar, lam, _ = KATALOG_PRODUK[p]
        # harga bervariasi (promo / varian) lalu dibulatkan ke kelipatan Rp500
        harga = harga_dasar * rng.normal(1.0, 0.07)
        harga = max(harga, harga_dasar * 0.7)
        harga = int(round(harga / 500) * 500)
        # jumlah terjual: hasil hitung bulat (Poisson) + pola musiman
        tumbuh = 1 + 0.35 * (t - hari[0]).days / len(hari)      # pertumbuhan bisnis ~35% / 2 th
        lam_eff = lam * _faktor_musiman(kat, t.month) * tumbuh
        jumlah = int(max(1, rng.poisson(lam_eff)))
        rows.append((p, kat, harga, jumlah, t))

    df = pd.DataFrame(rows, columns=["Produk", "Kategori", "Harga (Rp)",
                                     "Jumlah Terjual", "Tanggal"])
    return df


# ----------------------------------------------------------------------
# BEDAH KASUS 2 - Dataset Profil Mahasiswa
# ----------------------------------------------------------------------
JURUSAN = {
    # jurusan: (bobot, rata2 IPK)
    "Sistem Informasi":    (0.18, 3.35),
    "Teknik Informatika":  (0.17, 3.20),
    "Manajemen":           (0.17, 3.38),
    "Akuntansi":           (0.13, 3.30),
    "Psikologi":           (0.11, 3.45),
    "Ilmu Komunikasi":     (0.10, 3.40),
    "Teknik Sipil":        (0.08, 3.10),
    "Hukum":               (0.06, 3.25),
}


def buat_data_mahasiswa(n=1000, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    nama_j = list(JURUSAN)
    bobot_j = np.array([JURUSAN[j][0] for j in nama_j])
    bobot_j = bobot_j / bobot_j.sum()

    jurusan = rng.choice(nama_j, size=n, p=bobot_j)
    # semester 1-8 : mahasiswa semester awal lebih banyak (ada yang berhenti/lulus)
    bobot_s = np.array([0.17, 0.15, 0.15, 0.13, 0.13, 0.11, 0.09, 0.07])
    semester = rng.choice(np.arange(1, 9), size=n, p=bobot_s / bobot_s.sum())

    # Usia konsisten dengan semester: masuk kuliah ~18 th, naik 1 th per 2 semester
    usia = 18 + (semester - 1) // 2 + rng.choice([0, 0, 0, 1, 1, 2], size=n)
    senior = rng.random(n) < 0.03                     # mahasiswa kerja / usia lebih tua
    usia = np.where(senior, usia + rng.integers(3, 8, size=n), usia)
    usia = np.clip(usia, 17, 30)

    # IPK ~ normal per jurusan, sedikit naik seiring semester (seleksi alam), plus outlier rendah
    mu = np.array([JURUSAN[j][1] for j in jurusan]) + 0.015 * (semester - 4)
    ipk = rng.normal(mu, 0.30)
    rendah = rng.random(n) < 0.03
    ipk = np.where(rendah, rng.uniform(1.8, 2.5, size=n), ipk)
    ipk = np.round(np.clip(ipk, 1.8, 4.0), 2)

    return pd.DataFrame({"Jurusan": jurusan, "Semester": semester,
                         "IPK": ipk, "Usia": usia})


# ----------------------------------------------------------------------
# BEDAH KASUS 3 - Dataset Pengamatan Cuaca
# ----------------------------------------------------------------------
KOTA_CUACA = {
    # kota: (rata2 suhu, sd suhu, rata2 kelembaban)
    "Jakarta":    (31.5, 1.8, 74),
    "Bandung":    (25.0, 1.8, 82),
    "Surabaya":   (32.0, 1.6, 70),
    "Medan":      (31.0, 1.8, 76),
    "Makassar":   (31.5, 1.5, 76),
    "Yogyakarta": (29.5, 1.8, 77),
    "Semarang":   (31.0, 1.6, 75),
    "Denpasar":   (30.5, 1.4, 78),
    "Palembang":  (31.5, 1.6, 77),
    "Jayapura":   (31.0, 1.5, 80),
}
KONDISI = ["Cerah", "Berawan", "Hujan", "Hujan Lebat"]
# peluang kondisi cuaca berdasarkan tingkat kelembaban (batas bawah, probabilitas)
_PELUANG_KONDISI = [
    (0,  [0.80, 0.18, 0.02, 0.00]),
    (65, [0.55, 0.35, 0.10, 0.00]),
    (75, [0.25, 0.40, 0.30, 0.05]),
    (85, [0.08, 0.27, 0.50, 0.15]),
    (92, [0.02, 0.13, 0.55, 0.30]),
]


def buat_data_cuaca(n=1000, seed=SEED):
    rng = np.random.default_rng(seed + 2)
    kota_list = list(KOTA_CUACA)
    kota = rng.choice(kota_list, size=n)
    mu_s = np.array([KOTA_CUACA[k][0] for k in kota])
    sd_s = np.array([KOTA_CUACA[k][1] for k in kota])
    mu_h = np.array([KOTA_CUACA[k][2] for k in kota])

    suhu = rng.normal(mu_s, sd_s)
    # suhu makin panas -> kelembaban relatif turun (hubungan negatif)
    kelembaban = mu_h - 4.0 * (suhu - mu_s) + rng.normal(0, 6, size=n)
    kelembaban = np.clip(np.round(kelembaban), 40, 100).astype(int)
    suhu = np.round(suhu, 1)

    kondisi = []
    for h in kelembaban:
        peluang = [p for batas, p in _PELUANG_KONDISI if h >= batas][-1]
        kondisi.append(rng.choice(KONDISI, p=peluang))

    return pd.DataFrame({"Kota": kota, "Suhu (°C)": suhu,
                         "Kelembaban (%)": kelembaban, "Kondisi": kondisi})


def buat_semua_data(n=1000, seed=SEED):
    return (buat_data_penjualan(n, seed),
            buat_data_mahasiswa(n, seed),
            buat_data_cuaca(n, seed))


# ======================================================================
#  KONFIGURASI TAMPILAN
# ======================================================================
BIRU, ORANYE, ABU = "#3B9AE1", "#F29E38", "#7A8591"
PALET = px.colors.qualitative.Set2
NAMA_BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 4rem;}
    .hero {
        background: linear-gradient(120deg, #e8f4fd 0%, #fff4e3 100%);
        border: 1px solid #e3e8ee; border-radius: 16px;
        padding: 1.2rem 1.6rem; margin-bottom: 1rem;
    }
    .hero h1 {margin: 0 0 .2rem 0; font-size: 1.9rem; color: #1b2a3a;}
    .hero p  {margin: 0 0 .6rem 0; color: #4a5a6a;}
    .badge {
        display: inline-block; padding: .25rem .75rem; margin-right: .4rem;
        border-radius: 999px; background: #1b2a3a; color: #fff;
        font-size: .82rem; font-weight: 600;
    }
    .id-card {
        background: #1b2a3a; color: #fff; border-radius: 12px;
        padding: .8rem 1rem; margin-bottom: .6rem;
    }
    .id-card small {color: #9fb3c8;}
    .id-card b {font-size: 1.05rem;}
    .footer-id {
        position: fixed; left: 0; right: 0; bottom: 0; z-index: 999;
        text-align: center; padding: .35rem; font-size: .8rem;
        background: rgba(27,42,58,.92); color: #e6edf5;
    }
    .tipe {border-left: 5px solid; padding: .5rem .9rem; margin: .4rem 0;
           border-radius: 6px; background: #f7f9fb; color: #1b2a3a;}
    .tipe.biru {border-color: #3B9AE1;} .tipe.oranye {border-color: #F29E38;}
    .tipe.abu {border-color: #7A8591;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
#  FUNGSI BANTU
# ======================================================================
def rp(x):
    """Format rupiah ringkas: Rp 1,2 M / Rp 8,5 jt / Rp 75 rb."""
    if pd.isna(x):
        return "-"
    for batas, satuan in [(1e12, "T"), (1e9, "M"), (1e6, "jt"), (1e3, "rb")]:
        if abs(x) >= batas:
            return f"Rp {x / batas:,.1f} {satuan}".replace(".", ",")
    return f"Rp {x:,.0f}"


def tampil(fig, tinggi=None):
    """Tampilkan grafik Plotly (kompatibel dengan versi Streamlit lama & baru)."""
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=55, b=10),
                      title_font_size=16, legend_title_text="")
    if tinggi:
        fig.update_layout(height=tinggi)
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def tampil_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def header(judul, subjudul):
    st.markdown(
        f"""<div class="hero"><h1>{judul}</h1><p>{subjudul}</p>
        <span class="badge">👤 Nama: {NAMA}</span>
        <span class="badge">🆔 NIM: {NIM}</span></div>""",
        unsafe_allow_html=True,
    )


def kotak_tipe(warna, teks):
    st.markdown(f'<div class="tipe {warna}">{teks}</div>', unsafe_allow_html=True)


def ambil_rentang_tanggal(nilai, default):
    """st.date_input bisa mengembalikan 1 tanggal saat user baru memilih awal rentang."""
    if isinstance(nilai, (list, tuple)):
        if len(nilai) == 2:
            return nilai[0], nilai[1]
        if len(nilai) == 1:
            return nilai[0], nilai[0]
        return default
    return nilai, nilai


def ringkasan_numerik(df, kolom):
    """Tabel statistik deskriptif lengkap untuk beberapa kolom numerik."""
    out = {}
    for k in kolom:
        s = df[k].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        out[k] = {
            "Count": s.count(), "Mean": s.mean(), "Median": s.median(),
            "Std": s.std(), "Min": s.min(), "Q1": q1, "Q3": q3, "Max": s.max(),
            "IQR": q3 - q1, "Skewness": s.skew(), "CV (%)": s.std() / s.mean() * 100,
        }
    return pd.DataFrame(out).T


def jumlah_outlier_iqr(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    batas_bawah, batas_atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < batas_bawah) | (s > batas_atas)
    return mask, batas_bawah, batas_atas


@st.cache_data(show_spinner="Membangkitkan data sintetis...")
def muat_data(n, seed):
    return buat_semua_data(n=n, seed=seed)


# ======================================================================
#  HALAMAN: BERANDA
# ======================================================================
def halaman_beranda(d1, d2, d3):
    header("📊 Dashboard Bedah Kasus 1 – 3",
           "Mengenali tipe data (Nominal, Ordinal, Interval, Rasio, Waktu) "
           "dan memilih visualisasi yang tepat.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🛒 Kasus 1")
        st.caption("Dataset Operasional Penjualan")
        st.metric("Baris × Kolom", f"{d1.shape[0]:,} × {d1.shape[1]}")
        st.write("Produk, Kategori, Harga, Jumlah Terjual, Tanggal")
    with c2:
        st.subheader("🎓 Kasus 2")
        st.caption("Dataset Profil Mahasiswa")
        st.metric("Baris × Kolom", f"{d2.shape[0]:,} × {d2.shape[1]}")
        st.write("Jurusan, Semester, IPK, Usia")
    with c3:
        st.subheader("🌦️ Kasus 3")
        st.caption("Dataset Pengamatan Cuaca")
        st.metric("Baris × Kolom", f"{d3.shape[0]:,} × {d3.shape[1]}")
        st.write("Kota, Suhu, Kelembaban, Kondisi")

    st.markdown("### 🧭 Peta tipe data → visualisasi")
    peta = pd.DataFrame([
        ["Kasus 1", "Produk, Kategori", "Kategorik (Nominal)", "Bar Chart / Pie Chart"],
        ["Kasus 1", "Jumlah Terjual", "Numerik (Diskrit) – Rasio", "Histogram / Bar (hasil hitung bulat)"],
        ["Kasus 1", "Harga (Rp)", "Numerik (Kontinu) – Rasio", "Histogram"],
        ["Kasus 1", "Tanggal", "Waktu – Interval", "Line Chart (tren waktu)"],
        ["Kasus 2", "Jurusan", "Kategorik (Nominal)", "Bar Chart"],
        ["Kasus 2", "Semester", "Kategorik (Ordinal)", "Bar Chart berurutan"],
        ["Kasus 2", "IPK, Usia", "Numerik (Kontinu)", "Histogram / Box Plot"],
        ["Kasus 3", "Kota, Kondisi", "Kategorik (Nominal)", "Bar Chart / Pie Chart"],
        ["Kasus 3", "Suhu (°C)", "Numerik – Interval", "Histogram, Box Plot (rasio tidak berlaku)"],
        ["Kasus 3", "Kelembaban (%)", "Numerik – Rasio", "Histogram, Scatter (nol absolut)"],
    ], columns=["Kasus", "Kolom", "Tipe Data", "Visualisasi yang Sesuai"])
    tampil_df(peta, hide_index=True)

    kotak_tipe("biru", "🔵 <b>Nominal / Ordinal</b> — label kategori; ordinal punya urutan jelas.")
    kotak_tipe("oranye", "🟠 <b>Numerik (Interval / Rasio)</b> — bisa dihitung; rasio punya nol absolut.")
    kotak_tipe("abu", "⚪ <b>Waktu (Interval)</b> — jarak antar waktu bermakna, nol tidak absolut.")
    st.info("👈 Pilih halaman pada menu di sidebar. Setiap halaman punya filter interaktif, "
            "KPI, grafik, dan tabel data yang bisa diunduh.")


# ======================================================================
#  HALAMAN: KASUS 1 - PENJUALAN
# ======================================================================
def halaman_kasus1(df):
    header("🛒 Bedah Kasus 1: Dataset Operasional Penjualan",
           "Produk & Kategori (Nominal) · Harga (Kontinu-Rasio) · "
           "Jumlah Terjual (Diskrit-Rasio) · Tanggal (Waktu-Interval)")

    with st.sidebar:
        st.markdown("### 🔎 Filter — Kasus 1")
        semua_kat = sorted(df["Kategori"].unique())
        kat = st.multiselect("Kategori", semua_kat, default=semua_kat)
        opsi_produk = sorted(df[df["Kategori"].isin(kat)]["Produk"].unique())
        produk = st.multiselect("Produk", opsi_produk, default=opsi_produk)
        t_min, t_max = df["Tanggal"].min().date(), df["Tanggal"].max().date()
        rentang = st.date_input("Rentang tanggal", (t_min, t_max),
                                min_value=t_min, max_value=t_max)
        t0, t1 = ambil_rentang_tanggal(rentang, (t_min, t_max))
        h_min, h_max = int(df["Harga (Rp)"].min()), int(df["Harga (Rp)"].max())
        h0, h1 = st.slider("Rentang harga (Rp)", h_min, h_max, (h_min, h_max),
                           step=500, format="%d")

    m = (df["Kategori"].isin(kat) & df["Produk"].isin(produk)
         & (df["Tanggal"] >= pd.Timestamp(t0)) & (df["Tanggal"] <= pd.Timestamp(t1))
         & df["Harga (Rp)"].between(h0, h1))
    d = df[m].copy()
    if d.empty:
        st.warning("Tidak ada data yang cocok dengan filter. Longgarkan filter di sidebar.")
        return
    d["Pendapatan (Rp)"] = d["Harga (Rp)"] * d["Jumlah Terjual"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah transaksi", f"{len(d):,}")
    k2.metric("Total unit terjual", f"{d['Jumlah Terjual'].sum():,}")
    k3.metric("Total pendapatan", rp(d["Pendapatan (Rp)"].sum()))
    k4.metric("Rata-rata harga", rp(d["Harga (Rp)"].mean()))

    t_kat, t_num, t_waktu, t_data = st.tabs(
        ["🔵 Kategorik (Nominal)", "🟠 Numerik (Rasio)", "⚪ Waktu (Interval)", "📄 Data"])

    # ---------- Kategorik ----------
    with t_kat:
        kotak_tipe("biru", "<b>Produk & Kategori = Nominal</b> → bandingkan frekuensi dengan "
                           "<b>Bar Chart</b> (banyak kategori) atau <b>Pie Chart</b> (proporsi).")
        metrik = st.radio("Ukuran yang dibandingkan",
                          ["Frekuensi transaksi", "Total unit terjual", "Total pendapatan"],
                          horizontal=True)
        agg = (d.groupby(["Produk", "Kategori"], as_index=False)
                 .agg(**{"Frekuensi transaksi": ("Jumlah Terjual", "size"),
                         "Total unit terjual": ("Jumlah Terjual", "sum"),
                         "Total pendapatan": ("Pendapatan (Rp)", "sum")}))
        ca, cb = st.columns([1.5, 1])
        with ca:
            fig = px.bar(agg.sort_values(metrik), x=metrik, y="Produk", color="Kategori",
                         orientation="h", text_auto=".3s", color_discrete_sequence=PALET,
                         title=f"Bar Chart — {metrik} per Produk")
            tampil(fig, 560)
        with cb:
            per_kat = agg.groupby("Kategori", as_index=False)[metrik].sum()
            fig = px.pie(per_kat, names="Kategori", values=metrik, hole=0.0,
                         color_discrete_sequence=PALET,
                         title=f"Pie Chart — Proporsi {metrik} per Kategori")
            fig.update_traces(textinfo="percent+label")
            tampil(fig, 560)

    # ---------- Numerik ----------
    with t_num:
        kotak_tipe("oranye", "<b>Harga = Kontinu-Rasio</b> (punya nol absolut) → <b>Histogram</b>. "
                             "<b>Jumlah Terjual = Diskrit-Rasio</b> (hasil hitung bulat) → "
                             "histogram dengan bin bulat.")
        c1, c2 = st.columns(2)
        with c1:
            nb = st.slider("Jumlah bin — histogram harga", 10, 80, 30)
            logx = st.checkbox("Skala log (harga sangat beragam, disarankan)", value=True)
            warna = st.checkbox("Warnai per kategori", value=True)
            dh = d.copy()
            if logx:
                dh["Harga (log10)"] = np.log10(dh["Harga (Rp)"])
                fig = px.histogram(dh, x="Harga (log10)", nbins=nb,
                                   color="Kategori" if warna else None,
                                   color_discrete_sequence=PALET if warna else [ORANYE],
                                   title="Histogram Harga (skala log10)")
                kandidat = np.array([2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6, 2.5e6, 5e6, 1e7])
                lo, hi = dh["Harga (Rp)"].min(), dh["Harga (Rp)"].max()
                kandidat = kandidat[(kandidat >= lo * 0.9) & (kandidat <= hi * 1.1)]
                fig.update_xaxes(tickvals=np.log10(kandidat),
                                 ticktext=[rp(v) for v in kandidat], title_text="Harga (Rp)")
            else:
                fig = px.histogram(dh, x="Harga (Rp)", nbins=nb,
                                   color="Kategori" if warna else None,
                                   color_discrete_sequence=PALET if warna else [ORANYE],
                                   title="Histogram Harga")
            tampil(fig, 430)
        with c2:
            fig = px.box(d, x="Kategori", y="Harga (Rp)", color="Kategori", log_y=True,
                         color_discrete_sequence=PALET, points="outliers",
                         title="Box Plot Harga per Kategori (skala log)")
            fig.update_layout(showlegend=False)
            tampil(fig, 430 + 100)

        c3, c4 = st.columns(2)
        with c3:
            nb2 = st.slider("Jumlah bin — jumlah terjual", 10, 60, 25)
            fig = px.histogram(d, x="Jumlah Terjual", nbins=nb2, marginal="box",
                               color_discrete_sequence=[ORANYE],
                               title="Distribusi Jumlah Terjual (Diskrit)")
            tampil(fig, 430)
        with c4:
            st.markdown("**Statistik deskriptif**")
            tampil_df(ringkasan_numerik(d, ["Harga (Rp)", "Jumlah Terjual"]).round(2))
            rho = d[["Harga (Rp)", "Jumlah Terjual"]].corr(method="spearman").iloc[0, 1]
            st.metric("Korelasi Spearman (Harga vs Jumlah Terjual)", f"{rho:.2f}")
            st.caption("Negatif = produk lebih mahal cenderung terjual lebih sedikit per transaksi.")

    # ---------- Waktu ----------
    with t_waktu:
        kotak_tipe("abu", "<b>Tanggal = Waktu-Interval</b> → <b>Line Chart</b> untuk melihat tren waktu.")
        c1, c2, c3 = st.columns(3)
        gran = c1.radio("Granularitas", ["Harian", "Mingguan", "Bulanan"], index=2, horizontal=True)
        ukur = c2.selectbox("Ukuran", ["Total pendapatan", "Total unit terjual",
                                       "Jumlah transaksi", "Rata-rata harga"])
        pisah = c3.checkbox("Pisahkan per kategori", value=False)
        ma = c3.checkbox("Rata-rata bergerak", value=True, disabled=pisah)
        window = st.slider("Jendela rata-rata bergerak (periode)", 2, 12, 3, disabled=not ma or pisah)

        freq = {"Harian": "D", "Mingguan": "W", "Bulanan": "MS"}[gran]
        kolom, fungsi = {"Total pendapatan": ("Pendapatan (Rp)", "sum"),
                         "Total unit terjual": ("Jumlah Terjual", "sum"),
                         "Jumlah transaksi": ("Produk", "count"),
                         "Rata-rata harga": ("Harga (Rp)", "mean")}[ukur]
        if pisah:
            ts = (d.groupby([pd.Grouper(key="Tanggal", freq=freq), "Kategori"])[kolom]
                    .agg(fungsi).reset_index())
            fig = px.line(ts, x="Tanggal", y=kolom, color="Kategori", markers=True,
                          color_discrete_sequence=PALET,
                          title=f"Tren {ukur} per Kategori ({gran})")
        else:
            ts = d.groupby(pd.Grouper(key="Tanggal", freq=freq))[kolom].agg(fungsi).reset_index()
            fig = px.line(ts, x="Tanggal", y=kolom, markers=True,
                          color_discrete_sequence=[ABU], title=f"Tren {ukur} ({gran})")
            if ma:
                ts["MA"] = ts[kolom].rolling(window, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=ts["Tanggal"], y=ts["MA"], mode="lines",
                                         name=f"Rata-rata bergerak ({window})",
                                         line=dict(color=ORANYE, dash="dash", width=3)))
        fig.update_yaxes(title_text=ukur)
        tampil(fig, 450)

        st.markdown("**Pola musiman** — rata-rata unit terjual per transaksi (kategori × bulan)")
        piv = d.pivot_table(values="Jumlah Terjual", index="Kategori",
                            columns=d["Tanggal"].dt.month, aggfunc="mean")
        piv.columns = [NAMA_BULAN[int(b) - 1] for b in piv.columns]
        fig = px.imshow(piv, text_auto=".0f", aspect="auto", color_continuous_scale="Oranges",
                        labels=dict(color="Rata-rata unit"))
        tampil(fig, 380)

    # ---------- Data ----------
    with t_data:
        st.caption(f"{len(d):,} baris setelah filter.")
        tampil_df(d, hide_index=True)
        st.download_button("⬇️ Unduh CSV (hasil filter)", d.to_csv(index=False).encode("utf-8"),
                           "penjualan_filter.csv", "text/csv")


# ======================================================================
#  HALAMAN: KASUS 2 - MAHASISWA
# ======================================================================
def halaman_kasus2(df):
    header("🎓 Bedah Kasus 2: Dataset Profil Mahasiswa",
           "Jurusan (Nominal) · Semester (Ordinal) · IPK & Usia (Numerik Kontinu)")

    with st.sidebar:
        st.markdown("### 🔎 Filter — Kasus 2")
        semua_j = sorted(df["Jurusan"].unique())
        jur = st.multiselect("Jurusan", semua_j, default=semua_j)
        s0, s1 = st.select_slider("Semester", options=list(range(1, 9)), value=(1, 8))
        i0, i1 = st.slider("Rentang IPK", 0.0, 4.0, (float(df["IPK"].min()), 4.0), step=0.05)
        u_min, u_max = int(df["Usia"].min()), int(df["Usia"].max())
        u0, u1 = st.slider("Rentang usia", u_min, u_max, (u_min, u_max))

    m = (df["Jurusan"].isin(jur) & df["Semester"].between(s0, s1)
         & df["IPK"].between(i0, i1) & df["Usia"].between(u0, u1))
    d = df[m].copy()
    if d.empty:
        st.warning("Tidak ada data yang cocok dengan filter. Longgarkan filter di sidebar.")
        return

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Jumlah mahasiswa", f"{len(d):,}")
    k2.metric("Rata-rata IPK", f"{d['IPK'].mean():.2f}")
    k3.metric("Median IPK", f"{d['IPK'].median():.2f}")
    k4.metric("Rata-rata usia", f"{d['Usia'].mean():.1f} th")
    k5.metric("IPK ≥ 3,50", f"{(d['IPK'] >= 3.5).mean() * 100:.1f}%")

    t_kat, t_num, t_rel, t_data = st.tabs(
        ["🔵 Nominal & Ordinal", "🟠 Numerik (Kontinu)", "🔗 Hubungan Antar Variabel", "📄 Data"])

    # ---------- Kategorik ----------
    with t_kat:
        c1, c2 = st.columns(2)
        with c1:
            kotak_tipe("biru", "<b>Jurusan = Nominal</b> — hanya label pembeda, "
                               "urutkan berdasarkan frekuensi.")
            ft = d["Jurusan"].value_counts().reset_index()
            ft.columns = ["Jurusan", "Jumlah"]
            jenis = st.radio("Jenis grafik", ["Bar Chart", "Pie Chart"], horizontal=True)
            if jenis == "Bar Chart":
                fig = px.bar(ft.sort_values("Jumlah"), x="Jumlah", y="Jurusan", orientation="h",
                             text="Jumlah", color_discrete_sequence=[BIRU],
                             title="Jumlah Mahasiswa per Jurusan")
            else:
                fig = px.pie(ft, names="Jurusan", values="Jumlah",
                             color_discrete_sequence=PALET, title="Proporsi Mahasiswa per Jurusan")
                fig.update_traces(textinfo="percent+label")
            tampil(fig, 460)
        with c2:
            kotak_tipe("biru", "<b>Semester = Ordinal</b> — ada tingkatan 1 → 2 → 3; "
                               "tampilkan <b>Bar Chart berurutan</b> (bukan diurutkan frekuensi).")
            cnt = d["Semester"].value_counts().reindex(range(1, 9), fill_value=0)
            cum = cnt.cumsum() / cnt.sum() * 100
            lbl = [f"Sem {i}" for i in cnt.index]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=lbl, y=cnt.values, name="Jumlah", marker_color=BIRU,
                                 text=cnt.values, textposition="outside"), secondary_y=False)
            fig.add_trace(go.Scatter(x=lbl, y=cum.values, name="Kumulatif (%)",
                                     mode="lines+markers", line=dict(color=ORANYE, width=3)),
                          secondary_y=True)
            fig.update_yaxes(title_text="Jumlah mahasiswa", secondary_y=False)
            fig.update_yaxes(title_text="Kumulatif (%)", range=[0, 105], secondary_y=True)
            fig.update_layout(title="Distribusi Semester (berurutan) + Kumulatif")
            tampil(fig, 460 + 55)
            st.caption(f"Median semester = {int(d['Semester'].median())} · "
                       f"Modus = {int(d['Semester'].mode().iloc[0])} "
                       "(untuk ordinal gunakan median/modus, bukan rata-rata).")

    # ---------- Numerik ----------
    with t_num:
        kotak_tipe("oranye", "<b>IPK & Usia = Numerik Kontinu</b> → <b>Histogram</b> (distribusi) "
                             "atau <b>Box Plot</b> (sebaran & outlier).")
        var = st.radio("Variabel", ["IPK", "Usia"], horizontal=True)
        nb = st.slider("Jumlah bin", 5, 60, 25 if var == "IPK" else 12)
        s = d[var]
        c1, c2 = st.columns([1.6, 1])
        with c1:
            fig = px.histogram(d, x=var, nbins=nb, color_discrete_sequence=[ORANYE],
                               title=f"Histogram {var}")
            fig.add_vline(x=float(s.mean()), line_dash="dash", line_color="#1b2a3a",
                          annotation_text=f"Mean {s.mean():.2f}", annotation_position="top")
            fig.add_vline(x=float(s.median()), line_dash="dot", line_color=BIRU,
                          annotation_text=f"Median {s.median():.2f}", annotation_position="bottom")
            tampil(fig, 440)
        with c2:
            fig = px.box(d, y=var, points="outliers", color_discrete_sequence=[ORANYE],
                         title=f"Box Plot {var}")
            tampil(fig, 440)

        mask, bb, ba = jumlah_outlier_iqr(s)
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Skewness", f"{s.skew():.2f}")
        o2.metric("Batas bawah (IQR)", f"{bb:.2f}")
        o3.metric("Batas atas (IQR)", f"{ba:.2f}")
        o4.metric("Jumlah outlier", f"{int(mask.sum())}")
        st.markdown("**Statistik deskriptif**")
        tampil_df(ringkasan_numerik(d, ["IPK", "Usia"]).round(3))
        if mask.any():
            with st.expander(f"Lihat {int(mask.sum())} baris outlier {var}"):
                tampil_df(d[mask].sort_values(var), hide_index=True)

    # ---------- Hubungan ----------
    with t_rel:
        c1, c2 = st.columns(2)
        with c1:
            tmp = d.assign(Sem=d["Semester"].map(lambda v: f"Sem {v}"))
            fig = px.box(tmp, x="Sem", y="IPK", color_discrete_sequence=[BIRU],
                         category_orders={"Sem": [f"Sem {i}" for i in range(1, 9)]},
                         title="IPK per Semester (ordinal → urut)")
            fig.update_xaxes(title_text="Semester")
            tampil(fig, 430)
        with c2:
            urut = d.groupby("Jurusan")["IPK"].median().sort_values().index.tolist()
            fig = px.box(d, y="Jurusan", x="IPK", color="Jurusan", orientation="h",
                         category_orders={"Jurusan": urut}, color_discrete_sequence=PALET,
                         title="IPK per Jurusan (urut median)")
            fig.update_layout(showlegend=False)
            tampil(fig, 430)

        c3, c4 = st.columns(2)
        with c3:
            ct = pd.crosstab(d["Jurusan"], d["Semester"])
            fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                            title="Jumlah Mahasiswa: Jurusan × Semester")
            tampil(fig, 430)
        with c4:
            kor = d[["Semester", "IPK", "Usia"]].corr(method="spearman")
            fig = px.imshow(kor, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Korelasi Spearman (cocok untuk data ordinal)")
            tampil(fig, 430)

    # ---------- Data ----------
    with t_data:
        st.caption(f"{len(d):,} baris setelah filter.")
        tampil_df(d, hide_index=True)
        st.download_button("⬇️ Unduh CSV (hasil filter)", d.to_csv(index=False).encode("utf-8"),
                           "mahasiswa_filter.csv", "text/csv")


# ======================================================================
#  HALAMAN: KASUS 3 - CUACA
# ======================================================================
def halaman_kasus3(df):
    header("🌦️ Bedah Kasus 3: Dataset Pengamatan Cuaca",
           "Kota & Kondisi (Nominal) · Suhu (Interval) · Kelembaban (Rasio)")

    with st.sidebar:
        st.markdown("### 🔎 Filter — Kasus 3")
        semua_k = sorted(df["Kota"].unique())
        kota = st.multiselect("Kota", semua_k, default=semua_k)
        semua_c = [k for k in KONDISI if k in df["Kondisi"].unique()]
        kond = st.multiselect("Kondisi cuaca", semua_c, default=semua_c)
        s_min, s_max = float(df["Suhu (°C)"].min()), float(df["Suhu (°C)"].max())
        su0, su1 = st.slider("Rentang suhu (°C)", s_min, s_max, (s_min, s_max), step=0.1)
        h_min, h_max = int(df["Kelembaban (%)"].min()), int(df["Kelembaban (%)"].max())
        hu0, hu1 = st.slider("Rentang kelembaban (%)", h_min, h_max, (h_min, h_max))

    m = (df["Kota"].isin(kota) & df["Kondisi"].isin(kond)
         & df["Suhu (°C)"].between(su0, su1) & df["Kelembaban (%)"].between(hu0, hu1))
    d = df[m].copy()
    if d.empty:
        st.warning("Tidak ada data yang cocok dengan filter. Longgarkan filter di sidebar.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah observasi", f"{len(d):,}")
    k2.metric("Rata-rata suhu", f"{d['Suhu (°C)'].mean():.1f} °C")
    k3.metric("Rata-rata kelembaban", f"{d['Kelembaban (%)'].mean():.1f} %")
    k4.metric("Hari hujan", f"{d['Kondisi'].isin(['Hujan', 'Hujan Lebat']).mean() * 100:.1f}%")

    t_nom, t_int, t_rasio, t_rel, t_demo, t_data = st.tabs(
        ["🔵 Nominal", "🟠 Suhu (Interval)", "🟠 Kelembaban (Rasio)", "🔗 Hubungan",
         "🧪 Interval vs Rasio", "📄 Data"])

    # ---------- Nominal ----------
    with t_nom:
        kotak_tipe("biru", "<b>Kota & Kondisi = Nominal</b> — label teks tanpa urutan → "
                           "Bar Chart / Pie Chart.")
        c1, c2 = st.columns(2)
        with c1:
            ft = d["Kota"].value_counts().reset_index()
            ft.columns = ["Kota", "Jumlah"]
            fig = px.bar(ft.sort_values("Jumlah"), x="Jumlah", y="Kota", orientation="h",
                         text="Jumlah", color_discrete_sequence=[BIRU],
                         title="Jumlah Observasi per Kota")
            tampil(fig, 430)
        with c2:
            fk = d["Kondisi"].value_counts().reset_index()
            fk.columns = ["Kondisi", "Jumlah"]
            fig = px.pie(fk, names="Kondisi", values="Jumlah", color="Kondisi",
                         color_discrete_map={"Cerah": "#F6C85F", "Berawan": "#A0AEC0",
                                             "Hujan": BIRU, "Hujan Lebat": "#1F4E79"},
                         title="Proporsi Kondisi Cuaca")
            fig.update_traces(textinfo="percent+label")
            tampil(fig, 430)

    # ---------- Suhu ----------
    with t_int:
        kotak_tipe("oranye", "<b>Suhu (°C) = Interval</b> — jarak antar suhu sama, tetapi 0 °C "
                             "bukan 'tidak ada suhu'. Rasio (20 °C = 2× lebih panas dari 10 °C) "
                             "<b>tidak berlaku</b>.")
        nb = st.slider("Jumlah bin — suhu", 8, 60, 25)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(d, x="Suhu (°C)", nbins=nb, color_discrete_sequence=[ORANYE],
                               title="Histogram Suhu")
            fig.add_vline(x=float(d["Suhu (°C)"].mean()), line_dash="dash", line_color="#1b2a3a",
                          annotation_text=f"Mean {d['Suhu (°C)'].mean():.1f}")
            tampil(fig, 430)
        with c2:
            urut = d.groupby("Kota")["Suhu (°C)"].median().sort_values().index.tolist()
            fig = px.box(d, y="Kota", x="Suhu (°C)", color="Kota", orientation="h",
                         category_orders={"Kota": urut}, color_discrete_sequence=PALET,
                         title="Box Plot Suhu per Kota (urut median)")
            fig.update_layout(showlegend=False)
            tampil(fig, 430)

    # ---------- Kelembaban ----------
    with t_rasio:
        kotak_tipe("oranye", "<b>Kelembaban (%) = Rasio</b> — 0% benar-benar tidak ada kelembaban "
                             "(nol absolut), sehingga 80% = 2× lebih lembab dari 40%.")
        nb2 = st.slider("Jumlah bin — kelembaban", 8, 60, 20)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(d, x="Kelembaban (%)", nbins=nb2, color_discrete_sequence=[ORANYE],
                               title="Histogram Kelembaban")
            fig.add_vline(x=float(d["Kelembaban (%)"].mean()), line_dash="dash",
                          line_color="#1b2a3a",
                          annotation_text=f"Mean {d['Kelembaban (%)'].mean():.1f}")
            tampil(fig, 430)
        with c2:
            fig = px.box(d, x="Kondisi", y="Kelembaban (%)", color="Kondisi",
                         category_orders={"Kondisi": KONDISI},
                         color_discrete_map={"Cerah": "#F6C85F", "Berawan": "#A0AEC0",
                                             "Hujan": BIRU, "Hujan Lebat": "#1F4E79"},
                         title="Kelembaban menurut Kondisi Cuaca")
            fig.update_layout(showlegend=False)
            tampil(fig, 430)
        st.markdown("**Statistik deskriptif**")
        tampil_df(ringkasan_numerik(d, ["Suhu (°C)", "Kelembaban (%)"]).round(3))

    # ---------- Hubungan ----------
    with t_rel:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            fig = px.scatter(d, x="Suhu (°C)", y="Kelembaban (%)", color="Kondisi", opacity=0.7,
                             category_orders={"Kondisi": KONDISI},
                             color_discrete_map={"Cerah": "#F6C85F", "Berawan": "#A0AEC0",
                                                 "Hujan": BIRU, "Hujan Lebat": "#1F4E79"},
                             title="Suhu vs Kelembaban")
            if len(d) > 2 and d["Suhu (°C)"].nunique() > 1:
                koef = np.polyfit(d["Suhu (°C)"], d["Kelembaban (%)"], 1)
                xs = np.array([d["Suhu (°C)"].min(), d["Suhu (°C)"].max()])
                fig.add_trace(go.Scatter(x=xs, y=np.polyval(koef, xs), mode="lines",
                                         name="Garis regresi",
                                         line=dict(color="black", dash="dash")))
            tampil(fig, 480)
        with c2:
            r = d["Suhu (°C)"].corr(d["Kelembaban (%)"])
            st.metric("Korelasi Pearson (Suhu vs Kelembaban)", f"{r:.2f}" if pd.notna(r) else "-")
            st.caption("Negatif: semakin panas, kelembaban relatif cenderung turun.")
            ct = pd.crosstab(d["Kota"], d["Kondisi"], normalize="index") * 100
            ct = ct.reindex(columns=[k for k in KONDISI if k in ct.columns])
            fig = px.imshow(ct.round(1), text_auto=".0f", aspect="auto",
                            color_continuous_scale="Blues",
                            title="Kondisi Cuaca per Kota (% baris)")
            tampil(fig, 400)

    # ---------- Demo ----------
    with t_demo:
        st.markdown("#### 🧪 Mengapa rasio berlaku pada Kelembaban tetapi tidak pada Suhu °C?")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Suhu (Interval)**")
            a = st.number_input("Suhu A (°C)", value=10.0, step=1.0)
            b = st.number_input("Suhu B (°C)", value=20.0, step=1.0)
            r_c = "tidak terdefinisi" if a == 0 else f"{b / a:.2f}×"
            r_k = (b + 273.15) / (a + 273.15)
            st.metric("Rasio B/A dalam °C (menyesatkan)", r_c)
            st.metric("Rasio B/A dalam Kelvin (bermakna)", f"{r_k:.3f}×")
            st.info("Pada Kelvin (nol absolut) rasio baru bermakna. 20 °C hanya "
                    f"≈ {r_k:.3f}× lebih panas dari 10 °C, bukan 2×.")
        with c2:
            st.markdown("**Kelembaban (Rasio)**")
            x = st.slider("Kelembaban A (%)", 1, 100, 40)
            y = st.slider("Kelembaban B (%)", 1, 100, 80)
            st.metric("Rasio B/A", f"{y / x:.2f}×")
            st.success("Valid: kelembaban punya nol absolut, sehingga perbandingan "
                       "'2× lebih lembab' bermakna.")

    # ---------- Data ----------
    with t_data:
        st.caption(f"{len(d):,} baris setelah filter.")
        tampil_df(d, hide_index=True)
        st.download_button("⬇️ Unduh CSV (hasil filter)", d.to_csv(index=False).encode("utf-8"),
                           "cuaca_filter.csv", "text/csv")


# ======================================================================
#  MAIN
# ======================================================================
def main():
    with st.sidebar:
        st.markdown(
            f"""<div class="id-card"><small>MAHASISWA</small><br><b>👤 {NAMA}</b><br>
            <small>NIM</small> <b>{NIM}</b></div>""",
            unsafe_allow_html=True,
        )
        halaman = st.radio("📌 Navigasi", ["🏠 Beranda", "🛒 Kasus 1 — Penjualan",
                                          "🎓 Kasus 2 — Mahasiswa", "🌦️ Kasus 3 — Cuaca"])
        with st.expander("⚙️ Pengaturan data sintetis"):
            n = st.slider("Jumlah baris", 200, 5000, 1000, step=100)
            seed = st.number_input("Seed (acak)", min_value=0, max_value=2**31 - 1,
                                   value=SEED, step=1)
        st.divider()

    d1, d2, d3 = muat_data(int(n), int(seed))

    if halaman.startswith("🏠"):
        halaman_beranda(d1, d2, d3)
    elif "Kasus 1" in halaman:
        halaman_kasus1(d1)
    elif "Kasus 2" in halaman:
        halaman_kasus2(d2)
    else:
        halaman_kasus3(d3)

    st.markdown(f'<div class="footer-id">📊 Bedah Kasus 1–3 &nbsp;|&nbsp; '
                f'Nama: <b>{NAMA}</b> &nbsp;|&nbsp; NIM: <b>{NIM}</b></div>',
                unsafe_allow_html=True)


main()
