import joblib
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = 'model/worker_model.pkl'
bundle     = joblib.load(MODEL_PATH)
model      = bundle['model']
columns    = bundle['columns']
ac_avg     = bundle['agency_country_avg']
aj_avg     = bundle['agency_job_avg']

AGENCIES = {
    'Al Majd': {
        'arabic':    'المجد',
        'score':     4.8,
        'reviews':   1243,
        'speed':     5,
        'reliability': 5,
        'badge':     'الأسرع',
        'comments': [
            ('أحمد الشمري',    'خدمة ممتازة، وصلت العاملة في الوقت المحدد تماماً'),
            ('نورة العتيبي',   'تعامل راقي واحترافي، أنصح الجميع'),
            ('فهد القحطاني',   'أفضل وكالة تعاملت معها، سرعة في الإنجاز'),
        ]
    },
    'Al Rawabi': {
        'arabic':    'الروابي',
        'score':     4.5,
        'reviews':   987,
        'speed':     4,
        'reliability': 5,
        'badge':     'موثوق',
        'comments': [
            ('سارة المطيري',   'خدمة جيدة جداً وأسعار مناسبة'),
            ('خالد الدوسري',   'التزام تام بالمواعيد، شكراً للفريق'),
            ('منى الحربي',     'تجربة ممتازة، سأتعامل معهم مجدداً'),
        ]
    },
    'Barq': {
        'arabic':    'برق',
        'score':     4.2,
        'reviews':   654,
        'speed':     4,
        'reliability': 4,
        'badge':     'جيد',
        'comments': [
            ('عبدالله الزهراني','خدمة لا بأس بها، تأخر بسيط في الوصول'),
            ('ريم السبيعي',    'العاملة ممتازة، الوكالة متعاونة'),
            ('سلطان العنزي',   'تجربة مقبولة، يمكن التحسين في التواصل'),
        ]
    },
    'Sama': {
        'arabic':    'سما',
        'score':     3.9,
        'reviews':   432,
        'speed':     3,
        'reliability': 4,
        'badge':     'متوسط',
        'comments': [
            ('هند الرشيدي',    'الخدمة متوسطة، تأخر ملحوظ في الإجراءات'),
            ('بدر الشهري',     'يحتاج تطوير في التواصل مع العميل'),
            ('لطيفة الكندي',   'تجربة عادية، لا شيء مميز'),
        ]
    },
    'Nakheel': {
        'arabic':    'نخيل',
        'score':     3.7,
        'reviews':   321,
        'speed':     3,
        'reliability': 3,
        'badge':     'متوسط',
        'comments': [
            ('محمد الغامدي',   'الخدمة مقبولة لكن يوجد تأخير أحياناً'),
            ('أسماء البلوي',   'التواصل بطيء نوعاً ما'),
            ('يوسف المالكي',   'تجربة عادية، الأسعار معقولة'),
        ]
    },
    'Al Wafa': {
        'arabic':    'الوفاء',
        'score':     3.5,
        'reviews':   278,
        'speed':     3,
        'reliability': 3,
        'badge':     None,
        'comments': [
            ('نجلاء الحمدان',  'تأخر في الوصول عن الموعد المحدد'),
            ('وليد الصالح',    'الخدمة تحتاج تحسين كبير'),
            ('عزيزة العمري',   'لم تكن التجربة بالمستوى المطلوب'),
        ]
    },
    'Tadbeer': {
        'arabic':    'تدبير',
        'score':     4.6,
        'reviews':   1102,
        'speed':     5,
        'reliability': 4,
        'badge':     'سريع',
        'comments': [
            ('رانيا الجهني',   'سرعة في الإنجاز ودقة في المواعيد'),
            ('طارق العجمي',    'تجربة رائعة، وكالة محترمة جداً'),
            ('حصة المنصور',    'خدمة ممتازة وفريق متعاون'),
        ]
    },
    'Al Amal': {
        'arabic':    'الأمل',
        'score':     3.3,
        'reviews':   189,
        'speed':     2,
        'reliability': 3,
        'badge':     None,
        'comments': [
            ('سمية الخالدي',   'تأخير كبير في الإجراءات، غير راضية'),
            ('نادر السعدون',   'تجربة سيئة، لا أنصح بهم'),
            ('وفاء الرشيد',    'تواصل ضعيف وتأخير متكرر'),
        ]
    },
    'Rayhan': {
        'arabic':    'ريحان',
        'score':     3.1,
        'reviews':   143,
        'speed':     2,
        'reliability': 2,
        'badge':     None,
        'comments': [
            ('عمر الشريف',     'أبطأ وكالة تعاملت معها'),
            ('دانة المطلق',    'إجراءات طويلة جداً وتواصل سيء'),
            ('حمد البريكي',    'لن أتعامل معهم مجدداً'),
        ]
    },
}

COUNTRIES = {
    'Philippines': 'الفلبين',
    'Indonesia':   'إندونيسيا',
    'Sri Lanka':   'سريلانكا',
    'Ethiopia':    'إثيوبيا',
    'India':       'الهند',
    'Bangladesh':  'بنغلاديش',
}

JOBS = {
    'housemaid': 'عاملة منزلية',
    'driver':    'سائق',
}


@app.route('/')
def index():
    agencies = [{'key': k, 'arabic': v['arabic'], 'score': v['score']} for k, v in AGENCIES.items()]
    return render_template('index.html', agencies=agencies, countries=COUNTRIES, jobs=JOBS)


@app.route('/predict', methods=['POST'])
def predict():
    d = request.get_json()
    agency  = d.get('agency')
    country = d.get('country')
    job     = d.get('job')

    if agency not in AGENCIES or country not in COUNTRIES or job not in JOBS:
        return jsonify({'error': 'بيانات غير صحيحة'}), 400

    now = datetime.today()
    row = {
        'agency': agency, 'gender': 'F' if job == 'housemaid' else 'M',
        'country': country, 'job': job,
        'order_month': now.month, 'order_day_of_week': now.weekday(), 'order_hour': now.hour,
    }

    ac = ac_avg[(ac_avg['agency'] == agency) & (ac_avg['country'] == country)]
    aj = aj_avg[(aj_avg['agency'] == agency) & (aj_avg['job'] == job)]
    row['agency_country_avg'] = float(ac['agency_country_avg'].values[0]) if len(ac) else 50.0
    row['agency_job_avg']     = float(aj['agency_job_avg'].values[0])     if len(aj) else 50.0

    X    = pd.get_dummies(pd.DataFrame([row])).reindex(columns=columns, fill_value=0)
    days = int(round(model.predict(X)[0]))
    arrival = (now + timedelta(days=days)).strftime('%Y-%m-%d')

    info = AGENCIES[agency]
    return jsonify({
        'days':            days,
        'arrival':         arrival,
        'agency_arabic':   info['arabic'],
        'score':           info['score'],
        'reviews':         info['reviews'],
        'speed':           info['speed'],
        'reliability':     info['reliability'],
        'badge':           info['badge'],
        'comments':        info['comments'],
        'agency_avg':      round(row['agency_country_avg'], 1),
        'country_arabic':  COUNTRIES[country],
        'job_arabic':      JOBS[job],
    })


@app.route('/agencies')
def agencies():
    result = []
    for key, info in AGENCIES.items():
        result.append({
            'key': key, 'arabic': info['arabic'],
            'score': info['score'], 'reviews': info['reviews'],
            'speed': info['speed'], 'reliability': info['reliability'],
            'badge': info['badge'],
        })
    result.sort(key=lambda x: x['score'], reverse=True)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5002)
