from flask import Flask, request, render_template, session, redirect, url_for
from model import get_address_info

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parse_address/')
def parse_address():
    addr = request.args.get('addr', None)
    r = get_address_info(addr)
    for k in r:
        r[k] = ''.join(r[k])
    return r

if __name__=='__main__':
    app.run(host='0.0.0.0', port=1234)