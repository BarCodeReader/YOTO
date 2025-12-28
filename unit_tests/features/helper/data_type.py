from behave import register_type
import json

register_type(Json=lambda x: json.loads(x))
