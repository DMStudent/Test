# -*- coding: utf-8 -*-
# File : webTest.py
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 2018/12/4
#!/bin/bash


import web
import os
import time
urls = (
    '/(.*)', 'hello'
)

app = web.application(urls, globals())
app_root = os.path.dirname(__file__)
templates_root = os.path.join(app_root, 'templates')
render = web.template.render(templates_root)

class hello:
    def GET(self, name):
        if not name:
            name = 'World'
        # return '<h1>Hello, ' + name + '!</h1>'
        return render.reply_text(name)


if __name__ == "__main__":
    app.run()
