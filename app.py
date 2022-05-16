import logging
import time
import flask
import telebot
import os
import pandas as pd
import re
import joblib
import configparser
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
from sklearn.feature_extraction.text import TfidfVectorizer
import comment_parser
import visualize

config = configparser.RawConfigParser()
config.read('conf.ini')

logger = logging.getLogger('logger')
logger.setLevel(config.get('LOGGING', 'level'))
handler = RotatingFileHandler(config.get('LOGGING', 'filename'), maxBytes=1000000, backupCount=10)
formatter = Formatter(fmt='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', )
handler.setFormatter(formatter)
logger.addHandler(handler)

primary_data_en = config.get('FILES', 'source_en')
primary_data_ru = config.get('FILES', 'source_ru')  # каталог датасетов
stopwords_en = config.get('FILES', 'stopwords_en')
stopwords_ru = config.get('FILES', 'stopwords_ru')  # каталог стоп слов
model_en = config.get('FILES', 'model_en')
model_ru = config.get('FILES', 'model_ru')  # каталог расположения моделей
project_path = config.get('FILES', 'project_path')  # каталог проекта
pg_dsn = config.get('DB', 'pg_dsn')  # строка подключения к БД

API_TOKEN = config.get('KEYS', 'token')

WEBHOOK_HOST = config.get('HOST', 'host')
WEBHOOK_PORT = config.get('HOST', 'port')
WEBHOOK_LISTEN = config.get('HOST', 'opened-port')

WEBHOOK_SSL_CERT = config.get('CERT', 'certificate')  # Path to the ssl certificate
WEBHOOK_SSL_PRIV = config.get('CERT', 'private')  # Path to the ssl private key


try:
    engine = create_engine(pg_dsn)
except Exception as e:
    logger.error('cannot reach DB: {ex}'.format(ex=str(e)))
    logger.info('<---- Processing END')
    sys.exit(1)

results = sqlalchemy.text("select * from users")
allowed = []
for row in results:
    allowed.append(str(row['id']))



with open(stopwords_en, 'r') as fp:
    # считываем сразу весь файл
    data_en = list(fp.read())

with open(stopwords_ru, 'r') as fp:
    # считываем сразу весь файл
    data_ru = list(fp.read())

primary_df_en = pd.read_csv(primary_data_en)
features_en = primary_df_en.iloc[:, 10].values
labels_en = primary_df_en.iloc[:, 1].values

primary_df_ru = pd.read_csv(primary_data_ru)
features_ru = primary_df_ru.iloc[:, 3].values
labels_ru = primary_df_ru.iloc[:, 0].values


def data_cleansing_en(eng_str_list):
    processed_features = []
    for sentence in range(0, len(eng_str_list)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(eng_str_list[sentence]))
        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)
        # Converting to Lowercase
        processed_feature = processed_feature.lower()
        processed_features.append(processed_feature)
    return processed_features


def data_cleansing_ru(ru_str_list):
    processed_features = []
    for sentence in range(0, len(ru_str_list)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(ru_str_list[sentence]))
        # remove all single characters
        processed_feature = re.sub(r'\s+[а-яА-Я]\s+', ' ', processed_feature)
        # Remove single characters from the start
        processed_feature = re.sub(r'\^[а-яА-Я]\s+', ' ', processed_feature)
        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)
        # Converting to Lowercase
        processed_feature = processed_feature.lower()
        processed_features.append(processed_feature)
    return processed_features


def predict_nlp(item_list, model, cleansing, vector):
    test_features = vector.transform(cleansing(item_list)).toarray()
    predictions = model.predict(test_features)
    return predictions


vectorizer_en = TfidfVectorizer(min_df=1, max_df=0.8, stop_words=data_en)
vectorizer_ru = TfidfVectorizer(min_df=1, max_df=0.8, stop_words=data_ru)
processed_features_en = vectorizer_en.fit_transform(data_cleansing_en(features_en)).toarray()
processed_features_ru = vectorizer_ru.fit_transform(data_cleansing_ru(features_ru)).toarray()

ru_model = joblib.load(model_ru)
en_model = joblib.load(model_en)

api_key = config.get('KEYS', 'api')
TOKEN = config.get('KEYS', 'token')
my_id = config.get('KEYS', 'my_id')

bot = telebot.TeleBot(TOKEN, parse_mode=None)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Hello, to use this bot provide Youtube Video URL and type /analyze(ru\eng)")


WEBHOOK_URL_BASE = "https://%s:%s" % (WEBHOOK_HOST, WEBHOOK_PORT)
WEBHOOK_URL_PATH = "/%s/" % (API_TOKEN)

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

bot = telebot.TeleBot(API_TOKEN)

app = flask.Flask(__name__)


# Empty webserver index, return nothing, just http 200
@app.route('/', methods=['GET', 'HEAD'])
def index():
    return ''


# Process webhook calls
@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    if flask.request.headers.get('content-type') == 'application/json':
        json_string = flask.request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    else:
        flask.abort(403)

@bot.message_handler(commands=['analyze-ru'])
def analysys(message):
    logger.info('<---- Processing BEGIN RU')
    if message.chat.id not in allowed:
        bot.reply_to(message, 'Forbidden')
        logger.error('Forbidden user ' + str(message.chat.id) + "  tried to use the bot")
        return
    else:
        URL = message.text[11:].strip()
        # 'http://www.youtube.com/watch?v=ZFqlHhCNBOI'
        regex = re.compile(
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?(?P<id>[A-Za-z0-9\-=_]{11})')
        match = regex.match(URL)
        if not match:
            logger.error('incorrect URL request ' + URL)
            bot.reply_to(message, 'Incorrect URL\n' + URL)
            logger.info('<---- Processing END RU (with errors)')
            return
        logger.info('Starting to analyze video ... ' + URL)
        VIDEO_ID = match.group('id')
        try:
            comment_parser.scrapper(VIDEO_ID)
        except Exception as e:
            logger.error('cannot reach Youtube: {ex}'.format(ex=str(e)))
            bot.reply_to(message, 'Sorry, cannot reach Youtube: {ex}'.format(ex=str(e)))
            logger.info('<---- Processing END RU (with errors)')
            return
        url_df = pd.read_csv(data_for_test)
        most_liked_comment = url_df['likeCount'].idxmax()
        data_test = url_df.iloc[:, 3].values
        result = list(predict_nlp(data_test, ru_model, data_cleansing_ru, vectorizer_ru))
        pos_count = result.count('positive')
        neg_count = result.count('negative')
        neu_count = result.count('neutral')
        visualize.create_image([pos_count, neg_count, neu_count])
        photo = open(project_path + os.sep + 'visual.png', 'rb')
        analysys_descr = 'По результатам анализа выявлено следующее: \nПозитивных комментариев {} \nНегативных ' \
                         'комментариев {} \nНейтральных комментариев {} \nКомментарий с наибольшим числом лайков\n{' \
                         '}'.format(pos_count, neg_count, neu_count, url_df.at[most_liked_comment,
                                                                               'textDisplay'])
        bot.send_photo(message.chat.id, photo, caption=analysys_descr)

        logger.info('<---- Processing END RU')

bot.remove_webhook()

time.sleep(0.1)

bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH,
                certificate=open(WEBHOOK_SSL_CERT, 'r'))

app.run(host=WEBHOOK_LISTEN,
        port=WEBHOOK_PORT,
        ssl_context=(WEBHOOK_SSL_CERT, WEBHOOK_SSL_PRIV),
        debug=True)