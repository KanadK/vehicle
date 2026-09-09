FROM python:3.9

WORKDIR /app

ADD . /app
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Bake the NLTK corpora into the image so the container needs no network at
# runtime and the first request is not delayed by a ~26 MB download.
#
# NLTK_DATA is read into nltk.data.path at both build and run time. The
# downloader writes to the first directory on that path that already exists
# and is writable, so the directory has to be created before the download.
# nltk_setup.py is the same module app.py calls, so the image can never be
# baked with a different resource list than the one the app looks for.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p "$NLTK_DATA" && python nltk_setup.py

EXPOSE 5000

# Serve with gunicorn rather than `python app.py`. The __main__ block starts the
# Flask development server with debug=True, which exposes the Werkzeug console
# (arbitrary code execution) to anyone who can reach the container.
#
# Sync workers, no threads: each worker handles one request at a time, so the
# global pyplot state used by the chart routes is never shared concurrently.
# The long timeout covers /predict/app, which scrapes every review for an app
# before rendering.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
