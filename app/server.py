from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import csv
import StringIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://drive.google.com/uc?export=download&id=1L1ZjqLiwwSgflxWe165Z-w5Yg83BBzNV'
export_file_name = 'vehicles-model-2.pkl'

classes = ['passenger','forestry-other','passenger-dark' 'empty-dark', 'log-truck-loaded', 'log-truck-empty', 'log-truck-dark', 'empty', 'industrial-commercial' ]
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    content = await (data['file'].read())
    s = str(content, 'utf-8')
    data = StringIO(s)
    mkdir('Downloaded_Images')
    download_images(data, 'Downloaded_Images')
    path2 = Path('Downloaded_Images')
    data = ImageList.from_folder(path)
    learn = load_learner(path, export_file_name, test=data)
    y, _ = learn.get_preds(DatasetType.Test)
    y = torch.argmax(y, dim=1)
    preds = [learn.data.classes[int(x)] for x in y]
    # rm -r 'Downloaded_Images'
    resultsFile = open('results.csv', 'wb')
    wr = csv.writer(resultsFile)
    wr.writerows([preds])
    return FileResponse('results.csv')

# @app.route('/analyze', methods=['POST'])
# async def analyze(request):
#     data = await request.form()
#     img_bytes = await (data['file'].read())
#     img = open_image(BytesIO(img_bytes))
#     prediction = learn.predict(img)[0]
#     return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
