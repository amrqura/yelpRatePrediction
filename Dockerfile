FROM ubuntu:14.04

#ADD etc/apt/sources.list /etc/apt/sources.list

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y libatlas3-base libatlas-base-dev
RUN apt-get build-dep -y python3-numpy python3-scipy
RUN pip3 install numpy
RUN pip3 install gensim nltk==3.0.0b1
RUN pip3 install sklearn
#ADD requirements.txt /
#RUN pip3 install -r requirements.txt
# Some corpus for nltk
RUN python3 -m nltk.downloader -d /usr/share/nltk_data brown
RUN python3 -m nltk.downloader -d /usr/share/nltk_data punkt
RUN python3 -m nltk.downloader -d /usr/share/nltk_data words
RUN python3 -m nltk.downloader -d /usr/share/nltk_data stopwords
RUN python3 -m nltk.downloader -d /usr/share/nltk_data names
COPY TextClassification.py /src/TextClassification.py
COPY preprocssing.py /src/preprocssing.py
COPY ratings.txt /src/ratings.txt
COPY reviews.txt /src/reviews.txt
