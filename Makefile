all:	init

# Install dependencies
init:
	pip install -r requirements.txt

# Clean virtualenv folder
clean:
	rm -rv .env/
