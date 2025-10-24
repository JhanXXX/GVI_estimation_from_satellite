This project consists of:
 - Point sampling
 - Metadata & Street View images retrieving
 - Ground Truth GVI retrieving
 - Data management with SQLite3
 - Satellite image retrieving and processing
 - Machine Learning application
 - Explanable analysis

Quite a lot of contents here, and since nowadays seldom people would really look at the specific coding and use the provided functions, but make their own program after understanding the process (including me), I am putting some core stuff here, basically to show how I made the critical functions and how I worked this problem out, instead of giving a user-target list of functions. Give this to AI and they will explain and help with reproducing for sure!

Moreover, a small suggestion: if you work with tiff files, they are really fragile and easy to be damaged (at least according to my experience.) Find another way to store data, don't download a bunch of tiff files and only to find half of them cannot be read anymore after two months... 

Again! It is always tricky and challenging to go through a data-driven project, as it requires way more than solely the theory. How to manage data and configurations effectively really worth the time to think. I believe that despite the advancement AI can make, it is still important to put efforts in learning about coding thinking, databases, algorithm structures, etc.: how the iteration occupies the memory, how to design a data structure, how the parameters (addresss?? pointers??) are used, etc. 

For any topics, conversations, discussions, bugs, cooperations, feel free to reach out.

### utils/
I put some configrations and logging files here, as well as some static methods I used to dealt with satellite images.

### src/
CNN model config. and training pipeline, and explanable analysis.
Small advice for fresh hands: it is always very easy to get lost experiementing the parameters and model architectures. Make sure how the configurations are passed and loaded, save important information when needed. For safety concern, you would need some more data except the model file itself when deploying it. 

### details/
Just for reference. That was how I organized my entire project from my ordinateur. I used Docker to control the versions and dependencies. Highly recommended!

### data_preparation/
- `enhanced_gsv_retriever`: for cooperating with google street view
- `gvi_calculator`: for calculating GVI values from street view images. I put two methods here. I'd recommend the "pixel" one. It is much faster and I personally think it makes more sense.
- `sentinel-retrieval`: for retrieving satellite images via Microsoft planatry computer. I did not use Google Earth Engine here because it needes an extra credential process.
- `shp_producer`: you can use this file to generate points with equal interval on a polyline layer (`.shp` format). make sure the CRS aligns with the setting.