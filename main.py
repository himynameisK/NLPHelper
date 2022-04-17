import os
import pandas as pd
import re
import joblib
import configparser
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
from sklearn.feature_extraction.text import TfidfVectorizer
import telebot
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
primary_data_ru = config.get('FILES', 'source_ru')  # каталог для исходящих данных
stopwords_en = config.get('FILES', 'stopwords_en')
stopwords_ru = config.get('FILES', 'stopwords_ru')
model_en = config.get('FILES', 'model_en')
model_ru = config.get('FILES', 'model_ru')
data_for_test = config.get('FILES', 'data_for_test')
project_path = config.get('FILES', 'project_path')
pg_dsn = config.get('DB', 'pg_dsn')  # строка подключения к БД

allowed = [677169336]

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


@bot.message_handler(commands=['analyze-en'])
def analysys(message):
    logger.info('<---- Processing BEGIN EN')
    if message.chat.id not in allowed:
        bot.reply_to(message, 'Forbidden')
        logger.error('Forbidden user ' + str(message.chat.id) + "  tried to use the bot")
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
        logger.info('Starting to analyze video ... ' + URL)
        VIDEO_ID = match.group('id')
        print(VIDEO_ID)
        try:
            comment_parser.scrapper(VIDEO_ID)
        except Exception as e:
            logger.error('cannot reach Youtube: {ex}'.format(ex=str(e)))
            bot.reply_to(message, 'Sorry, cannot reach Youtube: {ex}'.format(ex=str(e)))
            logger.info('<---- Processing END EN (with errors)')
            return
        url_df = pd.read_csv(data_for_test)
        most_liked_comment = url_df['likeCount'].idxmax()
        data_test = url_df.iloc[:, 3].values
        result = list(predict_nlp(data_test, en_model, data_cleansing_en, vectorizer_en))
        pos_count = result.count('positive')
        neg_count = result.count('negative')
        neu_count = result.count('neutral')
        visualize.create_image([pos_count, neg_count, neu_count])
        photo = open(project_path + os.sep + 'visual.png', 'rb')
        analysys_descr = 'По результатам анализа выявлено следующее: \nПозитивных комментариев {} \nНегативных ' \
                         'комментариев {} \nНейтральных комментариев {} \nКомментарий с наибольшим числом лайков\n\n{' \
                         '}'.format(pos_count, neg_count, neu_count, url_df.at[most_liked_comment,
                                                                               'textDisplay'])
        bot.send_photo(message.chat.id, photo, caption=analysys_descr)
        logger.info('<---- Processing END EN')


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


bot.infinity_polling()
