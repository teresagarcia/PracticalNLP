# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import csv

class BookCsvPipeline():
    def open_spider(self, spider):
        self.file = open("output.csv", "at")
        fieldnames = ["title",
                    "author",
                    "price",
                    "stock",
                    "pages",
                    "binding",
                    "category",
                    "subcategory"]
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if self.file.tell() == 0:
            self.writer.writeheader()

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        self.writer.writerow(item)
        return item