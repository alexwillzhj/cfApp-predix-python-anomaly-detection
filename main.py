import os
import json
import redis
import requests
import cPickle as pickle
from flask import Flask, request
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import websocket
from apscheduler.schedulers.background import BackgroundScheduler
import code

app = Flask(__name__)

# Get VCAP_APPLICATION
cf_env = os.getenv('VCAP_APPLICATION')
if cf_env is None:
    # Config basic and UAA info
    with open('localConfig.json') as json_file:
        local_env = json.load(json_file)
    host = 'localhost'
    uaa_url = local_env['development']['uaa_url']
    base64ClientCredential = local_env['development']['base64ClientCredential']
    client_id = local_env['development']['client_id']
    grant_type = local_env['development']['grant_type']

    # Config redis
    redis_host = local_env['development']['redis_host']
    redis_port = int(local_env['development']['redis_port'])
    redis_password = local_env['development']['redis_password']
    redis_db = 0

    # Config Timeseries
    ts_query_url = local_env['development']['timeseries_query_url']
    ts_ingest_url = local_env['development']['timeseries_ingest_url']
    ts_zone_id = local_env['development']['timeseries_zone_id']

else:
    # Config basic and UAA info
    host = json.loads(cf_env)['application_uris'][0]
    uaa_url = str(os.getenv('uaa_url'))
    base64ClientCredential = str(os.getenv('base64ClientCredential'))
    client_id = str(os.getenv('client_id'))
    grant_type = str(os.getenv('grant_type'))

    # Config redis
    env_vars = os.environ.get('VCAP_SERVICES')
    redis_service = json.loads(env_vars)['redis'][0]
    redis_host = redis_service['credentials']['host']
    redis_port = redis_service['credentials']['port']
    redis_password = redis_service['credentials']['password']
    redis_db = 0

    # Config Timeseries
    ts_service = json.loads(env_vars)['predix-timeseries'][0]
    ts_query_url = ts_service['credentials']['query']['uri']
    ts_ingest_url = ts_service['credentials']['ingest']['uri']
    ts_zone_id = ts_service['credentials']['ingest']['zone-http-header-value']

# Obtain PORT
port = int(os.getenv("PORT", 64781))

# Initialize Timeseries info
ts = {'name': 'timeseries'}

# Initialize runtime_record
scheduler = BackgroundScheduler()


# Obtain TOKEN for Timeseries service
@app.route('/getToken', methods=['GET'])
def token_client():
    url = uaa_url + "/oauth/token"
    payload = "client_id=" + client_id + "&grant_type=" + grant_type
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'authorization': "Basic " + base64ClientCredential
    }
    response = requests.request("POST", url, data=payload, headers=headers)
    response_json = json.loads(response.text)
    return response_json['access_token']


@app.route('/getRedisStatus', methods=['GET'])
def setting_redis():
    try:
        app.r = redis.StrictRedis(host=redis_host,
                                  port=redis_port,
                                  password=redis_password,
                                  db=redis_db)
        return "Redis service ready."
    except:
        return "Redis service error."


@app.route('/')
def welcome_func():
    return "Welcome!"


@app.route('/saveModel')
def save_model_func(model_name="model_name_test", model_body="model_body_test"):
    app.r.set(model_name, model_body)
    model_get_body = app.r.get(model_name)
    return "Saved model: " + model_name + str(model_get_body)


@app.route('/getModel', methods=['GET'])
def get_model_func():
    model = 1
    return model


@app.route('/getTSStatus', methods=['GET'])
def setting_ts():
    # Get token
    ts['token'] = str(token_client())

    # Get times eries information in local mode
    try:
        ts['query_url'] = ts_query_url
        ts['ingest_url'] = ts_ingest_url
        ts['headers'] = {
            'content-type': "application/json",
            'authorization': "Bearer " + ts['token'],
            'Origin': 'https://www.predix.io',
            'predix-zone-id': ts_zone_id
        }
        return "Timeseries service ready."
    except:
        return "Timeseries service error."


@app.route('/tsQuery', methods=['GET'])
def ts_query_func(sample_number=10):

    # Get token or Update token
    setting_ts()

    # ts query body
    ts['query_body'] = '{"start": "1d-ago", "end": -1, "tags": \
                    [{"name": "ts_detection_feature0001","order": "desc","limit":' + str(sample_number) + '}, \
                    {"name": "ts_detection_label0001","order": "desc","limit":' + str(sample_number) + '}]}'

    response = requests.request("POST", ts['query_url'], data=ts['query_body'], headers=ts['headers'])

    return response.text


@app.route('/tsIngest', methods=['GET'])
def ts_ingest_func(ingest_tagname='ts_detection_res0001', ingest_timestamp='0', ingest_value='0'):

    # Get token or Update token
    setting_ts()

    # ts query body
    ts['ingest_body'] = '{"messageId": "1453338376222", \
                         "body": [{"name": "' + ingest_tagname + '",\
                         "datapoints":[[' + ingest_timestamp + ',' + ingest_value + ',3]], \
                                       "attributes": {"host": "server1","customer": "Acme"}}]}'

    ws = websocket.create_connection(ts['ingest_url'], header=ts['headers'])

    response = ws.send(ts['ingest_body'])

    return str(response)


@app.route('/trainModel', methods=['GET'])
def train_model_func():

    # get data from ts
    ts_data = ts_query_func(sample_number=1000)

    # code.interact(local=locals())

    # parse data
    data_json = json.loads(ts_data)

    # define feature
    data_feature = data_json['tags'][0]['results'][0]['values']
    data_feature_np = np.array(data_feature)[:, 1]
    data_feature_np = data_feature_np.reshape((data_feature_np.shape[0], 1))

    # define label
    data_label = data_json['tags'][1]['results'][0]['values']
    data_label_np = np.array(data_label)[:, 1]

    # train model
    model_qda = QuadraticDiscriminantAnalysis(store_covariances=False, priors=(0.9, 0.1))
    model_qda.fit(data_feature_np, data_label_np)

    # save model
    model_name = "model_qda"
    model_body = pickle.dumps(model_qda)
    save_model_func(model_name, model_body)

    return "model trained successfully: " + model_body


@app.route('/testModel', methods=['GET'])
def test_model_func():

    # get data from ts
    ts_data = ts_query_func(sample_number=1)

    # parse data
    data_json = json.loads(ts_data)

#    code.interact(local=locals())

    # define feature
    data_feature = data_json['tags'][0]['results'][0]['values']
    data_feature_sample = data_feature[0][1]

    data_timestamp = data_feature[0][0]

    # define model name
    model_name = "model_qda"

    # get model from redis
    model_body = app.r.get(model_name)

    # unpickle model
    model_qda = pickle.loads(model_body)

    # test last sample
    res = model_qda.predict(data_feature_sample)

    # ingest result to timeseries
    ts_ingest_func(ingest_tagname='ts_detection_res0001',
                   ingest_timestamp=str(data_timestamp),
                   ingest_value=str(res[0]))

    return "feature: " + str(data_feature_sample) + \
           ", classification result: " + str(res) + \
           "; ingest to ts: ingest_tagname (ts_detection_res0001)" + \
           ", timestamp (" + str(data_timestamp) + \
           "), value: (" + str(res[0]) + ')\n'


# start training job
job_train = scheduler.add_job(train_model_func, 'interval', hours=1)
job_train.pause()

# start testing job
job_test = scheduler.add_job(test_model_func, 'interval', seconds=10)
job_test.pause()


@app.route('/startTestingScheduler', methods=['GET'])
def start_testing_job_func():
    # start job
    global job_test
    job_test.resume()
    return "Job " + job_test.id + " started!"


@app.route('/pauseTestingScheduler', methods=['GET'])
def pause_testing_job_func():
    # pause job
    global job_test
    job_test.pause()
    return "Job " + job_test.id + " paused!"


@app.route('/startTrainingScheduler', methods=['GET'])
def start_training_job_func():
    # start job
    global job_train
    job_train.resume()
    return "Job " + job_train.id + " started!"


@app.route('/pauseTrainingScheduler', methods=['GET'])
def pause_training_job_func():
    # pause job
    global job_train
    job_train.pause()
    return "Job " + job_train.id + " paused!"


@app.route('/schedulerCondition', methods=['GET'])
def get_scheduler_condition_func():
    global scheduler_condition
    return scheduler_condition


if __name__ == '__main__':

    if cf_env is None:
        setting_ts()
        ts_query_func()
        ts_ingest_func()
        setting_redis()
        scheduler.start()

        app.run(host='127.0.0.1', port=5555)
    else:
        setting_ts()
        ts_query_func()
        ts_ingest_func()
        setting_redis()
        scheduler.start()

        app.run(host='0.0.0.0', port=port)
