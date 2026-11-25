
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Memuat model Random Forest
bundle = joblib.load('random_forest_co.pkl')

model = bundle['model']
feature_names = bundle['feature_names']
feature_medians = bundle['feature_medians']

# Nama fitur untuk tampilan aplikasi
feature_labels = {
    'PT08.S1(CO)': 'Sensor Karbon Monoksida',
    'C6H6(GT)': 'Konsentrasi Benzena',
    'PT08.S2(NMHC)': 'Sensor NMHC',
    'NOx(GT)': 'Nitrogen Oksida (NOx)',
    'PT08.S3(NOx)': 'Sensor NOx',
    'NO2(GT)': 'Nitrogen Dioksida (NO2)',
    'PT08.S4(NO2)': 'Sensor NO2',
    'PT08.S5(O3)': 'Sensor Ozon',
    'T': 'Suhu Udara',
    'RH': 'Kelembapan Relatif',
    'AH': 'Kelembapan Absolut',
    'Hour': 'Jam Pengukuran',
    'Day': 'Tanggal Pengukuran',
    'Month': 'Bulan Pengukuran',
    'DayOfWeek': 'Hari dalam Minggu'
}

# Penjelasan sederhana setiap fitur
feature_descriptions = {
    'PT08.S1(CO)': 'Nilai pembacaan sensor untuk mendeteksi karbon monoksida.',
    'C6H6(GT)': 'Perkiraan konsentrasi benzena di udara.',
    'PT08.S2(NMHC)': 'Nilai pembacaan sensor untuk senyawa NMHC.',
    'NOx(GT)': 'Konsentrasi nitrogen oksida di udara.',
    'PT08.S3(NOx)': 'Nilai pembacaan sensor untuk nitrogen oksida.',
    'NO2(GT)': 'Konsentrasi nitrogen dioksida di udara.',
    'PT08.S4(NO2)': 'Nilai pembacaan sensor yang berkaitan dengan NO2.',
    'PT08.S5(O3)': 'Nilai pembacaan sensor ozon.',
    'T': 'Suhu udara dalam derajat Celsius.',
    'RH': 'Persentase kelembapan udara.',
    'AH': 'Ukuran kelembapan absolut udara.',
    'Hour': 'Jam pengukuran, menggunakan format 0 sampai 23.',
    'Day': 'Tanggal pengukuran, dari 1 sampai 31.',
    'Month': 'Bulan pengukuran, dari 1 sampai 12.',
    'DayOfWeek': 'Hari pengukuran, 0 untuk Senin sampai 6 untuk Minggu.'
}

# Feature importance dari model
importance = sorted(
    zip(feature_names, model.feature_importances_),
    key=lambda item: item[1],
    reverse=True
)

top_features = [
    {
        'name': feature_labels.get(name, name),
        'value': round(float(value) * 100, 2)
    }
    for name, value in importance[:5]
]


@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None
    error = None
    mode = 'demo'

    # Keterangan hasil CO
    co_status = None
    co_message = None

    # Nilai awal menggunakan median dataset
    input_values = {
        name: float(feature_medians[name])
        for name in feature_names
    }

    if request.method == 'POST':

        mode = request.form.get('mode', 'manual')

        try:

            # MODE DEMO
            if mode == 'demo':
                pass

            # MODE MANUAL
            elif mode == 'manual':

                for name in feature_names:
                    value = request.form.get(name, '').strip()

                    if value == '':
                        raise ValueError(
                            f'{feature_labels.get(name, name)} belum diisi.'
                        )

                    input_values[name] = float(value)

            # Membuat DataFrame sesuai urutan fitur model
            input_df = pd.DataFrame(
                [[input_values[name] for name in feature_names]],
                columns=feature_names
            )

            # Melakukan prediksi
            prediction = float(model.predict(input_df)[0])

            if prediction < 0:
                prediction = 0.0

            # Membandingkan hasil dengan baku mutu CO 1 jam
            if prediction <= 10:
                co_status = 'Memenuhi Baku Mutu CO'
                co_message = (
                    'Prediksi konsentrasi CO berada di bawah '
                    'atau sama dengan batas baku mutu 1 jam.'
                )
            else:
                co_status = 'Melebihi Baku Mutu CO'
                co_message = (
                    'Prediksi konsentrasi CO melebihi '
                    'batas baku mutu 1 jam.'
                )

        except ValueError as e:
            error = str(e)

        except Exception:
            error = 'Terjadi kesalahan saat melakukan prediksi.'

    # Menyiapkan seluruh input untuk HTML
    fields = []

    for name in feature_names:
        fields.append({
            'name': name,
            'label': feature_labels.get(name, name),
            'description': feature_descriptions.get(name, ''),
            'value': input_values[name]
        })

    return render_template(
        'index.html',
        prediction=prediction,
        error=error,
        fields=fields,
        top_features=top_features,
        mode=mode,
        co_status=co_status,
        co_message=co_message
    )


if __name__ == '__main__':
    app.run(debug=True)
