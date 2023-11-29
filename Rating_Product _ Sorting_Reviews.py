

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayalım ve Var Olan Average Rating ile Kıyaslayalım.
###################################################

###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayalım.
###################################################

df = pd.read_csv("/Users/rmarabaci/PycharmProjects/pythonProject1/data_science_bootcamp/Hafta_5_Measurement_Problems/datasets/amazon_review.csv")
df["overall"].mean()


###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayalım.
###################################################

df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
current_date = df['reviewTime'].max()
df["day_diff"] = (current_date - df['reviewTime']).dt.days


q1, q2, q3 = df["day_diff"].quantile([.25,.50,.75])



def time_based_weighted_average(dataframe, q1, q2, q3, w1=50, w2=25, w3=15, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= q1, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > q1) & (dataframe["day_diff"] <= q2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > q2) & (dataframe["day_diff"] <= q3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > q3), "overall"].mean() * w4 / 100


time_based_weighted_average(df, q1, q2, q3)

df.loc[df["day_diff"] <= q1, "overall"].mean()
df.loc[(df["day_diff"] > q1) & (df["day_diff"] <= q2), "overall"].mean()
df.loc[(df["day_diff"] > q2) & (df["day_diff"] <= q3), "overall"].mean()
df.loc[(df["day_diff"] > q3), "overall"].mean()


###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyelim.
###################################################
###################################################
# Adım 1. helpful_no Değişkenini Üretelim
###################################################

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyelim.
###################################################

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

##################################################
# Adım 3. 20 Yorumu Belirleyelim.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)



