import scrapy

class BookSpider(scrapy.Spider):
    name = "books"

    start_urls = ["https://www.libreriageneral.es/"]

    def parse(self, response):
        num_cats_to_parse = 5
        cat_names = response.xpath("//div[@id='explore']/ul/li/a/text()").getall()
        cat_urls = response.xpath("//div[@id='explore']/ul/li/a/@href").getall()
        for _, name, url in zip(range(num_cats_to_parse), cat_names, cat_urls):
            name = name.strip()
            url = response.urljoin(url)
            yield scrapy.Request(url,
                                 callback=self.parse_category,
                                 cb_kwargs=dict(cat_name=name))

    def parse_category(self, response, cat_name):
        #de aquí saco subcategorías
        subcats_to_parse = 5
        subcat_names = response.xpath("//div[@class='bloque-top']/ul/li/a/text()").getall()
        subcat_urls = response.xpath("//div[@class='bloque-top']/ul/li/a/@href").getall()

        for _, name, url  in zip(range(subcats_to_parse),subcat_names, subcat_urls):
            name = name.strip()
            url = response.urljoin(url)
            yield scrapy.Request(url, callback=self.parse_subcategory,
                                 cb_kwargs=dict(cat_name=cat_name, subcat_name=name))

        # next_button = response.css(".next a")
        # if next_button:
        #     next_url = next_button.attrib["href"]
        #     next_url = response.urljoin(next_url)
        #     yield scrapy.Request(next_url,
        #                          callback=self.parse_category,
        #                          cb_kwargs=dict(cat_name=cat_name))
   
    def parse_subcategory(self, response, cat_name, subcat_name):    
        #de aquí saco los libros
        book_urls = response.xpath("//ul[contains(@class, 'listado_libros')]/li/form/dl/dd/a/@href").getall()[:10]

        for book_url in book_urls:
            book_url = response.urljoin(book_url)
            yield scrapy.Request(book_url, callback=self.parse_book,
                                 cb_kwargs=dict(cat_name=cat_name, subcat_name=subcat_name))

        # next_button = response.css(".next a")
        # if next_button:
        #     next_url = next_button.attrib["href"]
        #     next_url = response.urljoin(next_url)
        #     yield scrapy.Request(next_url,
        #                          callback=self.parse_category,
        #                          cb_kwargs=dict(cat_name=cat_name))

    def parse_book(self, response, cat_name, subcat_name):
        #los datos de los libros
        title = response.css(".summary h1::text").get()
        author = response.css("#autor a::text").get()
        price = response.xpath("//span[@itemprop='price']/text()").get()

        in_stock = response.xpath("//div[@class='disponibilidad']/span/text()").get()

        num_pages = response.xpath("//dd[@itemprop='numberOfPages']/text()").get()
        binding = response.xpath("//dd[@itemprop='bookFormat']/text()").get()

        yield {
            "title": title,
            "author": author,
            "price": price,
            "stock": in_stock,
            "pages": num_pages,
            "binding": binding,
            "category": cat_name,
            "subcategory": subcat_name
        }
