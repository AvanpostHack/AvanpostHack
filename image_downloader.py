from google_images_search import GoogleImagesSearch
from threading import Thread

def download_images(keyword, path_to_save, num=10):
    gis = GoogleImagesSearch('AIzaSyCpgNzZFXG3pVZo-jRRrsyZHpG-zPHkCEw', '63333369d1a0e48e7')
    _search_params = {
        'q': keyword,
        'num': num,
        'fileType': 'jpg|gif|png',
        # 'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived',
        # 'safe': 'active|high|medium|off|safeUndefined',  ##
        # 'imgType': 'clipart|face|lineart|stock|photo|animated|imgTypeUndefined',  ##
        # 'imgSize': 'huge|icon|large|medium|small|xlarge|xxlarge|imgSizeUndefined',  ##
        # 'imgDominantColor': 'black|blue|brown|gray|green|orange|pink|purple|red|teal|white|yellow|imgDominantColorUndefined',
        ##
        # 'imgColorType': 'color|gray|mono|trans|imgColorTypeUndefined'  ##
    }

    gis.search(search_params = _search_params, path_to_dir = path_to_save )

def start_image_downloads(keyword:str, path_to_save:str):
    thread = Thread(target = download_images, args = (keyword, path_to_save, 10))
    thread.start()
    print('donwload in progress ...')
    thread.join()
    print ('download ended')
