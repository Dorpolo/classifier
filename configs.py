import pandas as pd
import os

config_vars = {'feature_list': {
     'order_id': {
         'type': 'string',
         'model': False
         },
     'status': {
         'type': 'string',
         'model': False
         },
     'email_anoni': {
         'type': 'string',
         'model': True
         },
     'billing_country_code': {
         'type': 'string',
         'model': True
         },
     'shipping_country_code': {
         'type': 'string',
         'model': True
         },
     'shipping_method': {
         'type': 'string',
         'model': True
         },
     'created_at': {
         'type': 'timestamp',
         'model': False
         },
     'total_spent': {
         'type': 'numeric',
         'model': True
         },
     'currency_code': {
         'type': 'string',
         'model': True
        },
     'gateway': {
         'type': 'string',
         'model': True
         },
     'V1_link': {
         'type': 'boolean',
         'model': True
         },
     'V2_distance': {
         'type': 'numeric',
         'model': True
         },
     'V3_distance': {
         'type': 'numeric',
         'model': True
         },
     'V4_our_age': {
         'type': 'numeric',
         'model': True
         },
     'V5_merchant_age': {
         'type': 'numeric',
         'model': True
         },
     'V6_avs_result': {
         'type': 'string',
         'model': True
         },
     'V7_bill_ship_name_match': {
         'type': 'string',
         'model': True
         },
     'V8_ip': {
         'type': 'numeric',
         'model': True
         },
     'V9_cookie': {
         'type': 'numeric',
         'model': True
         },
     'V10_cookie': {
         'type': 'numeric',
         'model': True
         },
     'V11_cookie': {
         'type': 'numeric',
         'model': True
         }
     },
    'data': pd.read_csv(
            f'{os.getcwd()}/data/dataset.csv',
            low_memory=False
        )
    }
