pip install seaborn 
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
    st.title("Grafiksel Veri Analizi Raporu")

    # Veri setini yükleme
    @st.cache
    def load_data(file_path):
        df = pd.read_csv(file_path, delimiter=";")
        return df

    file_path = st.sidebar.file_uploader("Lütfen veri setini yükleyin", type=["csv"])

    if file_path is not None:
        df = load_data(file_path)

        st.subheader("İlk Beş Satır")
        st.write(df.head())

        st.subheader("Sütun Başlıkları")
        st.write(df.columns)

        df["Prim"] = pd.to_numeric(df["Prim"], errors="coerce")
        df["Teminat"] = pd.to_numeric(df["Teminat"], errors="coerce")

        df["PT_orani"] = df["Prim"] / df["Teminat"]
        df["PT_orani"] = (df["PT_orani"] - df["PT_orani"].min()) / (df["PT_orani"].max() - df["PT_orani"].min())

        df["PT_orani_transformed"] = np.log(df["PT_orani"] + 1)

        st.subheader(" PT_orani Dağılımı")
        plt.figure(figsize=(10, 6))
        sns.histplot(df["PT_orani_transformed"], kde=True, color="skyblue")
        plt.title("Transformed PT_orani Dağılımı")
        plt.xlabel("Transformed PT_orani")
        plt.ylabel("Frekans")
        st.pyplot()
        st.markdown("“PT_orani” adlı değişkenin dağılımını gösteriyor. Dağılımın sağa çarpık olduğunu söyleyebiliriz. Yani çoğu değer düşük, ancak bazı yüksek değerler de var. En yüksek frekansın olduğu bölge yaklaşık 0.1 civarında. Bu, veri setindeki “PT_orani” değerlerinin çoğunun bu aralıkta olduğunu gösteriyor. Değişkenin dağılımı oldukça geniş; minimum değerler 0’a yakın, maksimum değerler ise 0.5 civarında. Grafiğin sağ tarafında birkaç aykırı değer görünüyor. Bu noktalar, genel dağılımın dışında kalan istisnai değerler olabilir. "
        "Histogramın üzerine eklenen yoğunluk tahmini (KDE) sayesinde, dağılımın daha pürüzsüz bir şekilde nasıl dağıldığını görebiliriz.")
        st.subheader("Müşteri Sınıfına Göre PT_orani Kutu Grafiği")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Musteri_Sinifi", y="PT_orani", data=df)
        plt.title("Boxplot of PT_orani by Customer Class")
        plt.xlabel("Customer Class")
        plt.ylabel("PT_orani")
        st.pyplot()
        st.markdown("Müşteri sınıflarına göre Prim Teminat Oranı dağılımı incelendiğinde, neredeyse tüm sınıflarda benzer bir dağılım görülmektedir. Ancak, tüm sınıflarda aykırı değerler bulunmaktadır. Normal dağılıma en yakın olan sınıf, 4. sınıftır. Diğer sınıflar ise az da olsa sağa çarpıktır.Müşteri sınıflarının medyan değerleri birbirine oldukça yakındır. Bu nedenle sınıf düzeyleri belirlenirken son derece hassas olunmalıdır. Yaklaşık olarak medyan değerleri -5.2 olarak ifade edilebilir.")
        outlier_indices = []
        for customer_class in df["Musteri_Sinifi"].unique():
            class_data = df[df["Musteri_Sinifi"] == customer_class]
            q1 = class_data["PT_orani"].quantile(0.25)
            q3 = class_data["PT_orani"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_indices.extend(class_data[(class_data["PT_orani"] < lower_bound) | (
                        class_data["PT_orani"] > upper_bound)].index.tolist())

        df_filtered = df.drop(outlier_indices)
        st.subheader("Aykırı Değerler Kaldırıldıktan Sonra PT_orani Kutu Grafiği")
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_filtered, x="Musteri_Sinifi", y="PT_orani", hue="Musteri_Sinifi")
        plt.title("Box Plot of PT_orani Separated by Customer Classes (Outliers Dropped)")
        plt.xlabel("Customer Class")
        plt.ylabel("PT_orani")
        st.pyplot()
        st.markdown("Aykırı değerleri çıkardıktan sonra sağa çarpıklık daha belirgin hale geldi. Aynı zamanda minimum ve maksimum değerler arasındaki mesafe arttı.. Bu, dağılımın daha geniş bir aralığa yayıldığını gösteriyor.Kısmen aykırı değerlerin azalması da olumlu bir gelişme. Bu, veri setinin daha homojen hale geldiğini ve istatistiksel analizlerin daha güvenilir sonuçlar verebileceğini gösteriyor.")
        st.subheader("PairPlot ")
        sns.pairplot(df, vars=["Yas", "Gelir", "Prim", "Teminat", "PT_orani"])
        st.pyplot()
        st.markdown("Pairplot grafiği incelendiğinde, PT_orani ile Prim arasında belirgin bir doğrusal ilişki gözlemlenmektedir. Prim değeri arttıkça PT_orani de artmaktadır, bu da sigorta priminin teminat miktarına oranının arttığını gösterir. Bu doğrusal ilişki, Prim ile PT_orani arasında güçlü bir pozitif korelasyon olduğunu düşündürmektedir.Diğer değişkenler arasındaki ilişkiler konusunda daha belirgin bir yorum yapmak için, görsel olarak incelenen çiftlerin daha detaylı bir analizi gerekmektedir")

        st.subheader("Korelasyon Matrisi")

        corr = df[["Yas", "Gelir", "Prim", "Teminat", "PT_orani"]].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=0.5)
        plt.title("Korelasyon Matrisi")
        st.pyplot()
        st.markdown("Bu korelasyon matrisi, veri setindeki Yaş, Gelir, Prim, Teminat ve PT_orani değişkenleri arasındaki ilişkileri göstermektedir. Korelasyon katsayıları -1 ile 1 arasında değişir; 1 pozitif bir ilişkiyi, -1 ise negatif bir ilişkiyi, 0 ise ise herhangi bir ilişki olmadığını gösterir. Korelasyon matrisinde her değişkenin kendisiyle olan korelasyonu 1'dir (ana çapraz).PT_orani ile Teminat arasında -0.67 korelasyon katsayısı ile orta şiddetli bir negatif ilişki gözlemlenmektedir. Bu, bir değişkenin arttığında diğerinin genellikle azaldığını gösterir. Yani, genellikle prim oranı yüksek olan müşterilerin teminat miktarı daha düşüktür.Diğer tüm değişkenler arasındaki ilişkiler zayıftır. Örneğin, Yaş ile Gelir arasında pozitif bir ilişki gözlenirken, Prim ile Teminat arasında negatif bir ilişki gözlemlenir. Ancak bu ilişkiler, korelasyon katsayılarının düşük olması nedeniyle daha zayıftır.")




if __name__ == "__main__":
    main()
