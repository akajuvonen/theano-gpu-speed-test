all:	init

init:
	pip install -r requirements.txt

clean:
	rm -rv .env/
