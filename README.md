# Endüstriyel Atık Su Nötralizasyon Tanklarında Klasik Pekiştirmeli Öğrenme (Tabular Q-Learning & SARSA) ile Doğrusal Olmayan pH Dozajlama Kontrolü

**Yazar:** Nihad Nabiyev  
**Konu:** Ödev 11 — Endüstriyel Atık Su Nötralizasyon Tanklarında Doğrusal Olmayan pH Dozajlama Kontrolü  
**Yöntem Kısıtı:** Bu çalışmada yalnızca **klasik pekiştirmeli öğrenme** (Tabular Q-Learning ve Tabular SARSA) kullanılmıştır; derin pekiştirmeli öğrenme veya yapay sinir ağı (Recurrent PPO, LSTM, SAC, DQN vb.) **kullanılmamıştır**.

---

## Özet (Abstract)

Kimya, petrokimya ve tekstil boyama tesislerinin atık su arıtma ünitelerinde gerçekleştirilen $\text{pH}$ nötralizasyon prosesleri, kontrol mühendisliği literatüründeki en şiddetli doğrusal olmayan (non-linear) kontrol problemlerinden biridir. Güçlü asit-baz titrasyon eğrisinin eşdeğerlik noktası ($\text{pH} \approx 7.0$) civarındaki aşırı statik kazanç sıçraması ile karıştırma ve taşıma hatlarından kaynaklanan ölü zaman (time delay / dead-time) gecikmeleri, klasik sabit kazançlı kontrolcülerin (örneğin endüstriyel PID) sürekli aşım (overshoot) ve salınım (oscillation) üretmesine yol açmaktadır. Bu çalışmada, herhangi bir sinir ağı veya derin pekiştirmeli öğrenme yöntemi kullanılmaksızın, yalnızca **klasik tabular pekiştirmeli öğrenme** algoritmaları (**Off-Policy Tabular Q-Learning** ve **On-Policy Tabular SARSA**) ile çalışan uyarlanabilir bir kimyasal dozajlama kontrolcüsü geliştirilmiştir. Problemin boyut lanetini (curse of dimensionality) aşmak ve olaya tam fiziksel hakimiyet kurmak amacıyla; (i) titrasyon eğrisinin duyarlılığına uyarlı **eşitsiz aralıklı (non-uniform) durum ayrıklaştırma**, (ii) son $k$ adımdaki $\text{pH}$ sensör okumalarını, $\text{pH}$ değişim hızını, giriş suyu debisini, tank seviyesini ve ölü zaman tamponundaki geçmiş dozajlama eylemlerini birleştiren **Markov Karar Süreci (MDP)** kodlaması ve (iii) eşdeğerlik noktasında mikro-dozaj, şok anlarında hızlı tepki sağlayan **logaritmik ölçekli çift pompa eylem uzayı** tasarlanmıştır. Sürekli Karıştırmalı Tank Reaktörü (CSTR) simülasyon ortamında ani asidik ve bazik giriş şokları altında yapılan deneylerde, klasik RL ajanlarının klasik PID yaklaşımına kıyasla $\text{pH} \in [6.8, 7.2]$ yasal deşarj aralığında daha kararlı kaldığı, aşırı dozajlamayı azalttığı ve kimyasal tüketimini dengelediği gösterilmiştir.

---

## 1. Endüstriyel Problem ve Fiziksel Zorluklar

Kimya ve tekstil tesislerinin atık su arıtma tanklarında çıkış suyunun $\text{pH}$ değerinin çevre mevzuatı veya sonraki biyolojik arıtma kademeleri için belirlenen deşarj sınırları (örneğin $6.8 - 7.2$) içerisinde tutulması zorunludur. Ancak $\text{pH}$ nötralizasyon prosesi, endüstriyel otomasyon ve kontrol teorisinde şu temel zorlukları beraberinde getirir:

1. **Titrasyon Eğrisinin Şiddetli Doğrusal Olmayan Yapısı:** $\text{pH}$ tanımı hidrojen iyonu derişiminin negatif logaritması ($\text{pH} = -\log_{10}[H^+]$) olduğundan, eklenen asit veya baz miktarı ile çözeltinin $\text{pH}$ değeri arasındaki ilişki doğrusal değildir. Özellikle nötr bölge ($\text{pH} \approx 7.0$) civarında tamponlama kapasitesi çok düşüktür; bu nedenle eşdeğerlik noktasına yaklaşıldığında küçük dozaj değişimleri bile $\text{pH}$ değerinde çok büyük sıçramalara yol açar. Buna karşılık asidik veya bazik uç bölgelerde aynı miktarda reaktif ilavesi $\text{pH}$ değerini çok daha sınırlı değiştirir.
2. **Karıştırma ve Taşıma Ölü Zamanı (Time Delay):** Dozaj pompalarından asit veya baz enjekte edildiğinde, kimyasalın tank içerisinde homojen karışması ve sensör tarafından ölçülmesi anında gerçekleşmez; arada $k$ örnekleme adımlık ($\tau_d = k \cdot \Delta t$) bir ölü zaman bulunur. Geçmiş dozaj eylemlerini hesaba katmayan kontrolcüler, henüz etkisi sensöre ulaşmamış kimyasalın üzerine ek dozaj yaparak sistemi hedef bandın dışına taşırır ve sürekli aşım (overshoot) üretir.
3. **Değişken Giriş Debisi ve Bozucu Etki Şokları:** Tekstil boyama, yıkama ve kimyasal reaktör boşaltma hatlarında giriş suyunun debisi ($q_{\text{in}}$) ve asit/baz yükü zamanla değişir. Ani gelen bazik veya asidik şoklar tanktaki iyon dengesini hızla bozabilir.
4. **Klasik Pekiştirmeli Öğrenme Kısıtı ve Boyut Yönetimi:** Bu ödevde derin pekiştirmeli öğrenme yasak olduğundan, sinir ağı ağırlıklarıyla yüksek boyutlu sürekli uzayları örtmek yerine proses fiziğine dayalı akıllı durum kodlaması ve ayrıklaştırma ile klasik tabular RL yöntemlerinin etkin kullanılması gerekmektedir.

---

## 2. Kimyasal Proses Modeli (CSTR Dinamiği)

Simülasyon ortamı (`ph_env.py`), fiziksel kütle korunumu ve sulu çözelti iyon dengesi ilkelerine dayalı Sürekli Karıştırmalı Tank Reaktörü (CSTR) modeliyle kurulmuştur.

### 2.1. İyon Dengesi ve Analitik pH Hesabı

Güçlü asit ve güçlü baz iyonlarının bulunduğu sulu ortamda net iyon derişimi farkı $x = [B^+] - [A^-]$ ($\text{mol/L}$) ve suyun iyonlaşma sabiti $K_w = [H^+][OH^-] = 10^{-14}$ ($\text{mol}^2/\text{L}^2$, $25^\circ\text{C}$) olarak alındığında elektro-nötrlük bağıntısı:

$$[H^+] - [OH^-] + x = 0 \implies [H^+]^2 + x[H^+] - K_w = 0$$

Sayısal hesaplamalarda kayan nokta taşmalarını ve $\text{pH} \approx 7$ civarındaki hassasiyet kaybını önlemek için asidik ($x \le 0$) ve bazik ($x > 0$) bölgelerde analitik kökler ayrı ayrı ifade edilir:

$$\text{pH}(x) = \begin{cases} -\log_{10}\!\left(\dfrac{-x + \sqrt{x^2 + 4K_w}}{2}\right), & x \le 0 \\ 14 + \log_{10}\!\left(\dfrac{x + \sqrt{x^2 + 4K_w}}{2}\right), & x > 0 \end{cases}$$

### 2.2. Statik Kazanç Değişimi ve Kütle/Seviye Denklemleri

$x = \frac{K_w}{[H^+]} - [H^+]$ bağıntısının $\text{pH}$'a göre türevi alındığında statik proses kazancının $\text{pH}=7.0$ civarında uç bölgelere kıyasla binlerce kat arttığı görülür; bu da nötr bölgeye yaklaşırken kaba dozaj adımlarının neden aşım yarattığını doğrudan açıklar.

Kesit alanı $A$, sıvı seviyesi $h(t)$ ve reaktör hacmi $V(t) = 1000 \cdot A \cdot h(t)$ ($\text{L}$) olan tankta, asit ve baz dozaj pompalarının debi ayarları $u_a(t)$, $u_b(t)$ ve $k$-adımlık ölü zaman gecikmesi altında sistem dinamikleri şu diferansiyel denklemlerle modellenir:

$$\frac{dx(t)}{dt} = \frac{q_{\text{in}}(t)\big(x_{\text{in}}(t) - x(t)\big) + u_a(t - k\Delta t)\big(-C_a - x(t)\big) + u_b(t - k\Delta t)\big(C_b - x(t)\big)}{1000 \cdot A \cdot h(t)}$$

$$\frac{dh(t)}{dt} = \frac{q_{\text{in}}(t) + u_a(t - k\Delta t) + u_b(t - k\Delta t) - c_v\sqrt{h(t)}}{1000 \cdot A}$$

---

## 3. Markov Karar Süreci (MDP) Tasarımı

Ödev tanımındaki MDP gereksinimleri doğrultusunda durum, eylem ve ödül yapıları sinir ağı kullanılmadan doğrudan proses kontrolüne uyarlanmıştır:

* **Durum (State):** Son $k$ adımdaki $\text{pH}$ sensör okumaları ve $\text{pH}$ değişim hızı ($\Delta \text{pH}$), giriş suyunun debisi ($q_{\text{in}}$), tank seviyesi ($h$) ve ölü zaman tamponundaki geçmiş dozajlama eylemlerinin net toplamı ($\sum_{i=1}^{k} (u_{b,t-i} - u_{a,t-i})$) birleştirilir. Titrasyon eğrisinin dik olduğu $\text{pH} = 7.0$ civarında daha dar aralıklar, uzak bölgelerde ise daha geniş aralıklar kullanan **eşitsiz aralıklı ayrıklaştırma** uygulanır.
* **Eylem (Action):** Asit ve baz dozaj pompalarının debi ayarları ayrık kademeler halinde tanımlanır (kapalı, mikro dozaj, düşük, orta, yüksek ve maksimum asit/baz kademeleri). Eşdeğerlik noktasına yaklaşırken mikro dozaj kademeleri ince ayar sağlarken, ani şoklarda yüksek kademeler hızlı müdahale imkânı verir.
* **Ödül (Reward):** $\text{pH}$ değerinin hedef deşarj sınırları (örneğin $6.8 - 7.2$) içinde kalması pozitif ödül üretir. Sınır dışı deşarj durumlarında karesel ceza uygulanır; ayrıca harcanan kimyasal miktarı ve pompa debilerindeki ani değişimler (vana aşınması) negatif ceza olarak ödül fonksiyonuna eklenir.

---

## 4. Klasik Pekiştirmeli Öğrenme Yöntemleri

Derin RL yerine klasik Zamansal Fark ($\text{TD}(0)$) tabanlı tabular yöntemler kullanılmıştır:

1. **Tabular Q-Learning (Off-Policy):**
   $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

2. **Tabular SARSA (On-Policy):**
   $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

---

## 5. Deneysel Kurulum, PID Kıyaslaması ve Değerlendirme Metrikleri

Klasik pekiştirmeli öğrenme ajanlarının (Tabular Q-Learning ve Tabular SARSA) başarımını objektif biçimde ortaya koymak için, endüstride yaygın kullanılan klasik bir geri beslemeli **PID kontrolcü** referans (baseline) olarak alınmıştır. Ayrıca derin pekiştirmeli öğrenme yasağına tam uyum sağlamak için hiçbir sinir ağı veya fonksiyon yakınsama katmanı kullanılmamıştır.

### 5.1. Kontrol Senaryoları ve Bozucu Etkiler

Simülasyon ortamında (`ph_env.py`) zaman adımı $\Delta t = 5.0\text{ s}$ ve toplam bölüm uzunluğu $120$ adım ($600\text{ s}$) olarak tanımlanmıştır:
* **Başlangıç Koşulu:** Tank asidik veya bazik bir başlangıç $\text{pH}$ değerinden ve belirli bir sıvı seviyesinden ($h$) başlar.
* **Çoklu Bozucu Şokları:** Simülasyon süresince giriş suyunun debisi ($q_{\text{in}}$) ve iyonik derişimi ($x_{\text{in}}$) kademeli olarak değişir; örneğin başlangıçtaki asidik giriş rejiminin ardından ani bir **bazik tekstil atığı şoku** ve devamında **yüksek asidik şok** uygulanır.
* **Sensör Gürültüsü:** Gerçek endüstriyel $\text{pH}$ problarındaki ölçüm gürültüsünü modellemek için $\text{pH}$ okumalarına sıfır ortalamalı Gauss gürültüsü eklenir.

### 5.2. Performans Değerlendirme Metrikleri

Kontrolcülerin karşılaştırılmasında hem kontrol teorisi hem de endüstriyel proses işletmesi açısından şu metrikler kullanılır:
1. **Deşarj Limiti Uyumu (%):** Çıkış $\text{pH}$ değerinin hedef aralık ($6.8 \le \text{pH} \le 7.2$) içinde kaldığı adımların toplam adımlara oranıdır.
2. **İntegral Mutlak Hata (IAE):** Hedef $\text{pH}=7.0$ noktasından mutlak sapmanın zaman boyunca integralidir ($\text{IAE} = \sum_t \vert{}\text{pH}_t - 7.0\vert{}\,\Delta t$).
3. **İntegral Karesel Hata (ISE):** Büyük sapmaları ve aşırı dozaj kaynaklı sıçramaları daha güçlü cezalandıran metriktir ($\text{ISE} = \sum_t (\text{pH}_t - 7.0)^2\,\Delta t$).
4. **Maksimum Aşım (Overshoot):** Geçici başlangıç geçişi sonrasında hedeften oluşan en büyük sapma miktarıdır.
5. **Toplam Kimyasal Tüketimi ($\text{L}$):** Bölüm boyunca harcanan toplam asit ve baz dozaj hacmidir; işletme maliyetini doğrudan belirler.
6. **Vana Değişim Toplamı (Total Variation — TV):** Ardışık adımlardaki pompa debi değişimlerinin mutlak toplamıdır; aktüatör yıpranma eğilimini gösterir.

---

## 6. Deneysel Bulgular ve Fiziksel Yorum

### 6.1. Doğrusal Olmayan Bölgede Aşım ve Ölü Zaman Etkisi

Sabit kazançlı bir PID kontrolcü, eşdeğerlik noktası ($\text{pH} \approx 7.0$) civarındaki yüksek statik kazanç ve $k$-adımlık karıştırma gecikmesi nedeniyle dozaj komutunu zamanında kesemez. Henüz tankta yeterince karışmamış reaktifin etkisi sensöre yansımaya başladığında kontrolcü yön değiştirmekte gecikir; bu durum $\text{pH}$ değerinin hedef bandın üstüne veya altına taşmasına (overshoot) ve salınımlara yol açar.

### 6.2. Tabular Q-Learning ve SARSA Davranışı

* **Tabular Q-Learning:** Off-policy yapısı sayesinde her adımda ulaşılabilir en yüksek gelecek değerini ($\max_{a'} Q(s', a')$) hedef alarak ani şoklara karşı daha atak ve hızlı toparlanan bir politika öğrenir.
* **Tabular SARSA:** On-policy yapısı gereği $\epsilon$-greedy keşif politikasının getirdiği gerçek eylem geçişlerini güncellediği için özellikle eğitim süresince keşif kaynaklı dalgalanmalara karşı daha temkinli bir değer kestirimi üretir.
* **Eşitsiz Ayrıklaştırma ve Boru Hattı Hafızasının Katkısı:** Hedef deşarj bandı ($6.8 - 7.2$) çevresinde sıklaştırılmış $\text{pH}$ aralıkları ve geçmiş dozajların net toplamını içeren durum bilgisi, ajanın nötr noktaya yaklaşırken kademeli olarak mikro dozaj kademelerine inmesini ve ölü zaman kaynaklı aşımı azaltmasını sağlar.

---

## 7. Depo Yapısı ve Çalıştırma Adımları

```text
ph-tabular-rl/
├── README.md            # Makale formatında teknik dokümantasyon
├── requirements.txt     # Python bağımlılıkları (numpy, matplotlib)
├── ph_env.py            # Doğrusal olmayan CSTR pH nötralizasyon ortamı
├── agents.py            # Durum ayrıklaştırıcı, Q-Learning, SARSA ve PID ajanları
└── run_experiment.py    # Eğitim, değerlendirme, metrik hesabı ve grafik üretimi
   
