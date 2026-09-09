FROM python:3.9

WORKDIR /app

ADD . /app
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

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
