#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup

class DiscourseScraper:
    def __init__(self):
        # Configuration constants
        self.site_url = "https://discourse.onlinedegree.iitm.ac.in"
        self.target_category = 34
        self.auth_file = "session_data.json"
        self.output_folder = "scraped_topics"
        self.start_date = datetime(2025, 1, 1)
        self.end_date = datetime(2025, 4, 14)
        
        # Build API endpoints
        self.category_endpoint = f"{self.site_url}/c/courses/tds-kb/{self.target_category}.json"
    
    def convert_timestamp(self, timestamp_string):
        """Convert ISO timestamp to datetime object"""
        formats_to_try = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"]
        
        for fmt in formats_to_try:
            try:
                return datetime.strptime(timestamp_string, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse timestamp: {timestamp_string}")
    
    def perform_manual_authentication(self, pw_instance):
        """Handle user authentication through browser"""
        print("🚪 Authentication required - opening browser...")
        
        browser_instance = pw_instance.chromium.launch(headless=False)
        browser_context = browser_instance.new_context()
        current_page = browser_context.new_page()
        
        # Navigate to login page
        current_page.goto(f"{self.site_url}/login")
        print("🔑 Please complete Google authentication, then click Resume in Playwright toolbar")
        
        # Wait for user to complete login
        current_page.pause()
        
        # Store authentication state
        browser_context.storage_state(path=self.auth_file)
        print("💾 Authentication credentials saved")
        browser_instance.close()
    
    def verify_authentication_status(self, page_instance):
        """Check if current session is valid"""
        try:
            page_instance.goto(self.category_endpoint, timeout=10000)
            page_instance.wait_for_selector("pre", timeout=5000)
            
            # Try to parse JSON response
            response_text = page_instance.inner_text("pre")
            json.loads(response_text)
            return True
            
        except (PlaywrightTimeout, json.JSONDecodeError):
            return False
    
    def extract_all_topics(self, pw_instance):
        """Main scraping method to collect all topics"""
        print("🔍 Beginning topic extraction with saved session...")
        
        browser_instance = pw_instance.chromium.launch(headless=True)
        browser_context = browser_instance.new_context(storage_state=self.auth_file)
        page_instance = browser_context.new_page()
        
        collected_topics = []
        current_page = 0
        
        # Paginate through all topics
        while True:
            page_url = f"{self.category_endpoint}?page={current_page}"
            print(f"📋 Processing page {current_page}...")
            
            page_instance.goto(page_url)
            
            # Extract JSON data from page
            try:
                page_content = page_instance.inner_text("pre")
                parsed_data = json.loads(page_content)
            except:
                page_content = page_instance.content()
                parsed_data = json.loads(page_content)
            
            # Get topics from current page
            topic_list = parsed_data.get("topic_list", {}).get("topics", [])
            
            if not topic_list:
                print("📭 No more topics found - pagination complete")
                break
            
            collected_topics.extend(topic_list)
            current_page += 1
        
        print(f"📊 Total topics discovered: {len(collected_topics)}")
        
        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)
        files_saved = 0
        
        # Process each topic individually
        for topic_item in collected_topics:
            topic_creation_date = self.convert_timestamp(topic_item["created_at"])
            
            # Filter by date range
            if self.start_date <= topic_creation_date <= self.end_date:
                self._process_individual_topic(page_instance, topic_item)
                files_saved += 1
        
        print(f"✨ Successfully saved {files_saved} topic files to {self.output_folder}/")
        browser_instance.close()
    
    def _process_individual_topic(self, page_instance, topic_info):
        """Process and save individual topic data"""
        topic_json_url = f"{self.site_url}/t/{topic_info['slug']}/{topic_info['id']}.json"
        page_instance.goto(topic_json_url)
        
        # Extract topic data
        try:
            raw_content = page_instance.inner_text("pre")
            topic_details = json.loads(raw_content)
        except:
            raw_content = page_instance.content()
            topic_details = json.loads(raw_content)
        
        # Clean HTML content from posts
        post_list = topic_details.get("post_stream", {}).get("posts", [])
        for post_item in post_list:
            if "cooked" in post_item:
                # Strip HTML tags and get plain text
                soup = BeautifulSoup(post_item["cooked"], "html.parser")
                post_item["cooked"] = soup.get_text()
        
        # Generate filename and save
        output_filename = f"{topic_info['slug']}_{topic_info['id']}.json"
        output_path = os.path.join(self.output_folder, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(topic_details, output_file, indent=2, ensure_ascii=False)
    
    def execute_scraping_workflow(self):
        """Main execution method"""
        with sync_playwright() as playwright:
            # Check if authentication exists
            if not os.path.exists(self.auth_file):
                self.perform_manual_authentication(playwright)
            else:
                # Validate existing authentication
                browser_instance = playwright.chromium.launch(headless=True)
                browser_context = browser_instance.new_context(storage_state=self.auth_file)
                test_page = browser_context.new_page()
                
                if not self.verify_authentication_status(test_page):
                    print("⚠️ Stored session expired - reauthenticating...")
                    browser_instance.close()
                    self.perform_manual_authentication(playwright)
                else:
                    print("✅ Existing session validated successfully")
                    browser_instance.close()
            
            # Execute the scraping process
            self.extract_all_topics(playwright)

def run_scraper():
    """Entry point function"""
    print("🚀 Discourse Topic Scraper Starting...")
    scraper_instance = DiscourseScraper()
    scraper_instance.execute_scraping_workflow()
    print("🎉 Scraping process completed!")

if __name__ == "__main__":
    run_scraper()