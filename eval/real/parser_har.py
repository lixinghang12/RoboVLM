import json
import requests
import argparse

def parser_har(package_path: str, save_path: str):
    with open(package_path, 'r', encoding='utf-8') as f:
        har_data = json.load(f)

    # get the entry info
    entries = har_data['log']['entries']
    assert len(entries) == 1, "Note that the number of entry you download must be one"
    for entry in entries:
        request = entry['request']
        url = request['url']
        method = request['method']
        headers = {header['name'].replace(":", ""): header['value'] for header in request['headers']}

        # test to seed the request again with the info
        assert method == 'GET', "The port can only preserve get method request"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, "The response status must be the 200"
        with open(save_path, 'w') as file:
            json.dump(dict(
                url=url,
                headers=headers
            ), file)

if __name__ == "__main__":
    # set args by argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--package_path', type=str, help="the network har you download")
    parser.add_argument('--save_path', type=str, default='./result.json', help="the path you save the headers and url get from the network package path")
    args = parser.parse_args()
    parser_har(args.package_path, args.save_path)
