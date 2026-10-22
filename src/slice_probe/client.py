"""Small standard-library client for the optional private SLICE API."""
import json
from urllib.request import Request, urlopen


class Client:
    def __init__(self, url, api_key, *, timeout=120):
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def request(self, method, path, value=None, *, idempotency_key=None):
        headers = {'Authorization': 'Bearer ' + self.api_key}
        if value is not None:
            headers['Content-Type'] = 'application/json'
        if idempotency_key is not None:
            headers['Idempotency-Key'] = idempotency_key
        request = Request(self.url + path, method=method, headers=headers,
                          data=json.dumps(value).encode() if value is not None else None)
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def create_training_job(self, settings, *, idempotency_key):
        """Keep the key across retries, including after a client process restart."""
        return self.request('POST', '/v1/training-jobs', settings,
                            idempotency_key=idempotency_key)

    def job(self, identifier):
        return self.request('GET', '/v1/jobs/' + identifier)

    def cancel(self, identifier, *, idempotency_key):
        return self.request('POST', '/v1/jobs/' + identifier + '/cancel',
                            idempotency_key=idempotency_key)

    def score_batch(self, critic_version, states, *, return_calibrated_probability=False):
        return self.request('POST', '/v1/scoring-batches', dict(
            critic_version=critic_version, states=states,
            return_calibrated_probability=return_calibrated_probability))['results']
