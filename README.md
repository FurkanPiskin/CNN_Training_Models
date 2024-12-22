# CNN_Training_Models
 Animal  Kaggle Dataset Model Training
Proje Özeti: Evcil Hayvan Görsel Sınıflandırma Modeli
Bu proje, evcil hayvan sınıflarını tanıyabilen bir derin öğrenme modelinin geliştirilmesini amaçlamaktadır. Proje, Evcil Hayvan Görselleri (Animals with Attributes 2) veri kümesinden alınan resimlerle eğitilen bir konvolüsyonel sinir ağı (CNN) modeli kullanmaktadır. Model, çeşitli görüntü işleme teknikleri ve derin öğrenme metotlarıyla sınıflandırma görevini yerine getirir.

Adım 1: Veri Yükleme ve Ön İşleme
Veri Kümesi: Proje, 10 farklı evcil hayvan türünden oluşan bir veri kümesi kullanmaktadır. Bu sınıflar arasında collie, dolphin, elephant, fox, moose, rabbit, sheep, squirrel, giant panda ve polar bear yer alır.
Veri Yükleme: Veriler, her sınıfın alt dizinlerinden alınan 650 görsel ile sınırlıdır. Bu görsellerin her biri 64x64 piksel boyutlarına küçültülür.
Ön İşleme: Resimler, modelin eğitimi için uygun hale getirilmek amacıyla:
Normalizasyon: Piksel değerleri 0 ile 1 arasında normalize edilmiştir.
Yeniden Boyutlandırma: Görseller 64x64 boyutlarına yeniden boyutlandırılmıştır.
Adım 2: Renk Sabitleme (Gray World Algoritması)
Modelin doğru renk dengesini öğrenebilmesi için Gray World algoritması uygulanmıştır. Bu algoritma, her bir renk kanalının (Kırmızı, Yeşil, Mavi) ortalama değerini global bir ortalamaya çekerek görseldeki renk dengesizliklerini düzeltir.



def get_wb_images(img):
    mean_b = np.mean(img[:, :, 0])  # Blue kanal ortalaması
    mean_g = np.mean(img[:, :, 1])  # Green kanal ortalaması
    mean_r = np.mean(img[:, :, 2])  # Red kanal ortalaması
    mean_gray = (mean_b + mean_g + mean_r) / 3  # Global ortalama
    scale_b = mean_gray / mean_b
    scale_g = mean_gray / mean_g
    scale_r = mean_gray / mean_r
    img[:, :, 0] = np.clip(img[:, :, 0] * scale_b, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * scale_g, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * scale_r, 0, 255)
    return img.astype(np.uint8)
Adım 3: Veri Artırma (Data Augmentation)
Eğitim verisi üzerinde veri artırma işlemi uygulanmıştır. Bu, modelin genelleme yeteneğini artırmak için çeşitli görsel manipülasyon tekniklerini içerir. Bu teknikler arasında:

Yükseklik ve genişlik kaydırma (width_shift_range, height_shift_range),
Büyütme ve küçültme (zoom_range),
Dönme (rotation_range),
Yatay çevirme (horizontal_flip) gibi işlemler yer alır.
Bu işlemler, modelin daha fazla ve çeşitlendirilmiş veriye erişmesini sağlayarak overfitting'i engellemeye yardımcı olur.



dataGen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
dataGen.fit(x_train)

Etiketleme: Görsellerin etiketleri (sınıf adları), sayısal bir formatta kodlanarak modelin anlayacağı bir biçime getirilmiştir.

One-Hot Encoding: Etiketler, One-Hot Encoding kullanılarak kategorik bir formata dönüştürülmüştür. Bu, modelin sınıflandırma görevini daha etkili yapabilmesini sağlar.


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_train = to_categorical(y_train_encoded, nOfClasses)
Adım 5: Modelin Oluşturulması
Konvolüsyonel Sinir Ağı (CNN):

İlk Katman: 32 filtreli bir konvolüsyon katmanı ve ardından bir max-pooling katmanı.
İkinci Katman: 64 filtreli bir konvolüsyon katmanı ve max-pooling katmanı.
Flatten Katmanı: Özellik haritalarını düzleştirir, ardından dense katmanına bağlanır.
Fully Connected Katman: 128 nöronlu bir katman ve dropout (overfitting'i azaltmak için).
Çıktı Katmanı: Softmax aktivasyonu ile nOfClasses kadar sınıf için çıktı.

model = Sequential()
model.add(Conv2D(input_shape=(64, 64, 3), filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=nOfClasses, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
Adım 6: Model Eğitimi ve Değerlendirilmesi
Model, batch size ve epoch parametreleri ile eğitim verilmiştir. Eğitim sırasında doğrulama verisi de kullanılarak modelin performansı izlenmiştir.

Eğitim süreci sonrasında, modelin test verisi üzerindeki başarısı değerlendirilmiş ve kayıp ve doğruluk metrikleri hesaplanmıştır.


hist = model.fit(dataGen.flow(x_train, y_train, batch_size=64), epochs=20)
score = model.evaluate(x_test, y_test, verbose=1)
Adım 7: Sonuçlar
Eğitim süreci sonunda, modelin kayıp ve doğruluk değerleri çizilerek modelin nasıl öğrenmeye devam ettiğini görselleştirmiştir.
Son olarak, test verisi ile yapılan değerlendirme sonucunda, modelin performansı ölçülmüştür. Bu, modelin gerçek dünya verilerinde nasıl performans göstereceğini anlamak için önemlidir.
Kaydedilen Model
Eğitilen model .h5 formatında kaydedilmiştir. Bu, modelin daha sonra tekrar kullanılabilmesi veya başka bir ortamda test edilmesi için faydalıdır.

model.save("/kaggle/working/my_model10.h5")
Sonuç
Bu proje, evcil hayvan türlerini tanımak için kullanılan bir derin öğrenme modelinin geliştirilmesini ve bu modelin eğitilmesini içermektedir. Modelin başarısını artırmak için görüntü işleme teknikleri (renk sabitleme, veri artırma) ve derin öğrenme stratejileri (CNN, dropout, One-Hot Encoding) kullanılmıştır. Proje, görüntü sınıflandırma görevlerinde derin öğrenmenin etkin kullanımını göstermektedir.

Modelin performansını artırmak için şu adımları uygulayabilirsiniz:

Veri Kümesi Geliştirme:

Daha fazla görsel ve sınıf ekleyerek çeşitliliği artırın.
Veri dengesizliği varsa, eksik sınıflar için veri artırma uygulayın.
Veri Ön İşleme ve Artırma:

Gelişmiş veri artırma teknikleri (gürültü ekleme, ışık değişiklikleri, kesme).
Renk uzayı çeşitlendirme (HSV, LAB).
Model İyileştirme:

Daha derin CNN mimarileri ve farklı aktivasyon fonksiyonları deneyin.
Hiperparametre optimizasyonu (batch size, learning rate).
Transfer Learning:

VGG16, ResNet gibi önceden eğitilmiş modellerden yararlanın.
Regularization (Düzenleme):

Dropout, batch normalization ve L2 regularization ekleyin.
Eğitim Stratejileri:

Learning rate scheduler ve cross-validation kullanarak modelin doğruluğunu artırın.
Değerlendirme:

Confusion matrix, precision, recall ve F1 score gibi metrikleri analiz edin.
Bu yöntemler, modelin doğruluğunu artırırken overfitting’i önlemenize yardımcı olacaktır.
