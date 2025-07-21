"""
Flask application for Resume Roaster
------------------------------------

This module sets up a small Flask application that serves the frontend
from a template and exposes an API status endpoint.  If you have
additional API routes (such as `/api/analyze`, `/api/gallery`, etc.)
defined elsewhere, they can be registered using blueprints or by
importing them into this module.

The root route (`/`) renders the `index.html` file from the
`templates` directory.  The `/api/status` endpoint returns a JSON
diagnostic with information about available API routes and the
operational status of the service.
"""

from flask import Flask, render_template, jsonify

# Create the Flask application instance.  By default, Flask will
# search for templates in a folder named ``templates`` located in the
# same directory as this file.
app = Flask(__name__)


@app.route("/api/status")
def api_status() -> "jsonify":
    """Return a diagnostic JSON describing the API status.

    This endpoint lists the main API routes that the service exposes,
    as well as basic metadata like the service name, version and
    operational status.  Clients can use this endpoint to perform
    health checks or discover available API capabilities.

    Returns:
        flask.Response: A JSON response containing the API description.
    """
    return jsonify({
        "endpoints": {
            "/api/analysis/<id>": "Get specific analysis",
            "/api/gallery": "Get recent public analyses",
            "/api/stats": "Get platform statistics",
            "/health": "Health check",
            "/api/analyze": "Analyze a resume"
        },
        "service": "Resume Roaster API",
        "status": "operational",
        "version": "2.0"
    })


@app.route("/")
def home() -> "str":
    """Render the main landing page.

    When a client visits the root URL, this function renders the
    ``index.html`` template located in the ``templates`` directory.

    Returns:
        str: The rendered HTML content of the index page.
    """
    return render_template("index.html")


if __name__ == "__main__":
    # Run the Flask development server.  When deploying in production,
    # use a proper WSGI server (e.g., Gunicorn or uWSGI) instead.
    app.run(debug=True, host="0.0.0.0", port=5000)
