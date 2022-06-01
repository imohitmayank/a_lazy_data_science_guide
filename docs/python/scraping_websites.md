# Scraping websites using Scrapy

## Introduction 

- Scraping is the processing of traversing the web and collecting data from the web pages. People scrap websites for lots of reasons -- be it to get information about company or latest news or stock prices or just create a dataset for the next big AI model :sunglasses: 
- In this article, we will be using [Scrapy](https://docs.scrapy.org) to scrape the data from the [Devgan](http://devgan.in/all_sections_ipc.php) website that hosts details of difference sections in Indian Penal Code. I think this example provides the perfect blend of basic and advanced knowledge of scraping. So let's get started! 

!!! Tip
    The complete code is also available on [GitHub](https://github.com/imohitmayank/ipc_semantic_search/tree/main/devganscrap)

## Understanding the website

- Before we even start scraping, we need to understand the structure of the website. This is very important, as we want to (1) get an idea of what we want to scrap and (2) where those data are located.

<figure markdown> 
    ![](../imgs/scrapy_devgan.png)
    <figcaption>The flow of scraping section descriptions from Devgan website</figcaption>
</figure>

- Our goal is to scrap the description for each section in IPC. As obvious from the website flow above, the complete process can be divided into two parts, 
  - First, we need to traverse to the main page and extract the link of each sections.
  - Next, we need to traverse to each section and extract the description details present there.


## Data extraction methods

- Now, let's also look into different methods exposed by Scrapy to extract data from the web pages. The basic idea is that scrapy downloads the web page source code in HTML and parse it using different parsers. For this, we can either use `XPaths` or `CSS` selectors. The choice is purely up to us.
- To begin with, let's try to find the link of each section. For this, you can open inspect page from your browser by right clicking on any of the sections and select `inspect` option. This should show you the source code. Next, try to find the position the tag where section is defined. Refer the image below, and you can see that each section is within `<a>` tag inside the `<div id="content">` tag. The href component will give you the link of the section and the `<span>` tag inside gives the section name. 
  
<figure markdown> 
    ![](../imgs/scrapy_devgan_inspect.png)
    <figcaption>Inspecting the source code of the first relevant page in Devgan website</figcaption>
</figure>

- To try out the extraction, we can utilize the interpreter functionlaity of scapy. For that just activate your Scrpay VM and type `scrapy shell '{website-link}'`, here it will be `scrapy shell http://devgan.in/all_sections_ipc.php`. This opens a playground where we can play around with `response` variable to experiment what is extracted with different queries.
- To extract the individual sections we can use `CSS` query - `response.css('div#content').css('a')`. Note, here we define the `{tag_type}#{id}` as the CSS selector and then use another `CSS` query - `a`. This will give us a list of all the `<a>` tags inside the `<div id="content">` tag.
- Now from within section, to extract the title, we can use `CSS` query - `section.css('span.sectionlink::text').extract()`. For this to work, you should save the last query as `section` variable.
- Similar approach can be applied to extract the description from the next page. Just re-run the shell with one of the section's link and try out building `CSS` query. Once you have all the queries ready, we can move on to the main coding part :peace:


## Setup Scrapy project

- First, let us install the Scapy package using pip. It can be easily done by running the following command in the terminal: `pip install scrapy`. Do make sure to create your own virtual environment (VE), activate it and then install the package in that environment. For confusing regarding VE, refer my [snippets](http://mohitmayank.com/a_lazy_data_science_guide/python/python_snippets/#conda-cheat-sheet) on the same topic. 
- Next, let us setup the Scrapy project. Go to your directory of choice and run following command `scrapy startproject tutorial`. This will create a project with following folder structure,

``` shell
tutorial/
    scrapy.cfg            # deploy configuration file
    tutorial/             # project's Python module, you'll import your code from here
        __init__.py
        items.py          # project items definition file
        middlewares.py    # project middlewares file
        pipelines.py      # project pipelines file
        settings.py       # project settings file
        spiders/          # a directory where you'll later put your spiders
            __init__.py
``` 

!!! Note
    The above folder structure is taken from the [Scrapy Official Tutorial](https://docs.scrapy.org/en/latest/intro/tutorial.html#)


## Create your Spider

- Usually we create one spider to scrap one website. For this example we will do exactly the same for Devgan website. So let's create a spider `spiders/devgan.py`. The code is shown below, 

``` python linenums="1"
# import
import scrapy

# function
class QuotesSpider(scrapy.Spider):
    name = "devgan"
    allowed_domains = ["devgan.in"]

    def start_requests(self):
        urls = [
            'http://devgan.in/all_sections_ipc.php',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse_mainpage)

    def parse_mainpage(self, response):
        # identify the links to the individual section pages
        sections = response.css('div#content').css('a')#.getall()
        # for each section
        for section in sections:
            # loc var
            loc = {
                'title' : section.xpath('@title').extract(),
                'link' : 'http://devgan.in' + section.xpath('@href').extract()[0],
                'section': section.css('span.sectionlink::text').extract(),
            }
            # traverse again and extract the description
            yield scrapy.Request(loc['link'], callback=self.parse_section, 
                    cb_kwargs=dict(meta=loc))

    def parse_section(self, response, meta):
        # extract the description
        meta['description'] = " ".join(response.css('tr.mys-desc').css('::text').extract())
        # return
        return meta
```

- Now lets us try to understand the code line by line, 
  - `Line 2:` Importing the `scrapy` package.
  - `Line 5:` Defining the spider class that inherits the `scrapy.Spider` class.
  - `Line 6-7:` We define the name of the spider and allow the spider to crawl the domain `devgan.in`.
  - `Line 9-14:` We define the `start_requests` method. This method is called when the spider is started. It calls the scraping function for each url to scrap, it is done using `scrapy.Request` function call. For each url, the scraping function set within `callback` parameter is called. 
  - `Line 16:` We define the scraping function `parse_mainpage` for the main page. Note, this function receives `response` as an argument, that contains the response from the serverf for the main page url. 
  - `Line 18-29:` We start with identifying the links to the individual section pages, store them in `sections`. and calls the scraping function `parse_section` for each section. Before that, we also extract the title, link and section name from the main page using the queries we created before. One point to remember, this particular example is little complex as you want to traverse even inside into the section page. For this, we again call the `scrapy.Request` for the individual section link. Finally, we want to pass the data collected form this page to the section page, as we will consolidate all data for individual section there and return it. For this, we use `cb_kwargs` parameter to pass the meta data to the next function. 
  - `Line 31-35:` We extract the description from the section page using the `CSS` query. We add description detail to the metadata and return the complete data to be persisted.


## Executing the spider

- To run the spider, traverse to the root directory and execure the following command, `scrapy crawl devgan -O sections_details.csv -t csv`. Here, `devgan` is the name of the spider we created earlier, `-O` is used to set the output file name as `sections_details.csv`. `-t` is used to define the output format as `csv`. This will create the csv file with all details of the sections as separate columns!

And that's it! Cheers! :smile: