

build-image:
	docker build -t python_benchmarks:latest -f docker.run .;

run-image:
	docker run -t -v /"$$(pwd)"/results:/results python_benchmarks:latest
