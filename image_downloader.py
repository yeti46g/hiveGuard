import requests
import urllib

photo_url = []

for i in [1,2]:
    response = requests.get("https://www.inaturalist.org/observations.json?
    taxon_id=322284&page={}&per_page=200".format(i))
    photo_url += [e['photos'][0]['medium_url'] for e in response.json()]

# download images given urls

for i in range(len(photo_url)):
    urllib.request.urlretrieve(photo_url[i],
                        "hornet/{}.jpg".format(i))
