# Makefile for automating the deployment of a Flask application with Prometheus and Grafana

# Define variables
DOCKER_NETWORK = monitoring
FLASK_APP_NAME = flask_app
PROMETHEUS_CONTAINER = prometheus
GRAFANA_CONTAINER = grafana
PROMETHEUS_PORT = 9090
GRAFANA_PORT = 3000
FLASK_PORT = 9696
NUM_REQUESTS = 5

# Default target
all: create_network start_prometheus start_grafana build_flask_app start_flask_app simulate_requests

# Create Docker network if it doesn't exist
create_network:
	@if [ -z "$$(docker network ls --filter name=$(DOCKER_NETWORK) -q)" ]; then \
		docker network create $(DOCKER_NETWORK); \
	else \
		echo "Docker network $(DOCKER_NETWORK) already exists"; \
	fi

# Start Prometheus container
start_prometheus:
	-docker rm -f $(PROMETHEUS_CONTAINER)
	docker run -d --name=$(PROMETHEUS_CONTAINER) --network=$(DOCKER_NETWORK) -p $(PROMETHEUS_PORT):$(PROMETHEUS_PORT) -v $(PWD)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

# Start Grafana container
start_grafana:
	-docker rm -f $(GRAFANA_CONTAINER)
	docker run -d --name=$(GRAFANA_CONTAINER) --network=$(DOCKER_NETWORK) -p $(GRAFANA_PORT):$(GRAFANA_PORT) grafana/grafana

# Build Flask app Docker image
build_flask_app:
	-docker rm -f $(FLASK_APP_NAME)
	docker build -t $(FLASK_APP_NAME) .

# Start Flask app container
start_flask_app:
	docker run -d --name=$(FLASK_APP_NAME) --network=$(DOCKER_NETWORK) -p $(FLASK_PORT):$(FLASK_PORT) $(FLASK_APP_NAME)

# Simulate user requests
simulate_requests:
	python test_multiple_requests.py --num_requests $(NUM_REQUESTS)

# Clean up containers and network
clean:
	-docker rm -f $(PROMETHEUS_CONTAINER) $(GRAFANA_CONTAINER) $(FLASK_APP_NAME)
	-docker network rm $(DOCKER_NETWORK)

.PHONY: all create_network start_prometheus start_grafana build_flask_app start_flask_app simulate_requests clean