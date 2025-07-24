#!/usr/bin/env python3
"""
Match Data Tool for EGA.xlsx and Azure Vector Index

This tool compares data from EGA.xlsx with vector data stored in Azure Search index,
matching on location and showing both xlsx data and vector data for each row.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Any
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import json


class MatchDataTool:
    """Tool to match Excel data with Azure vector search data"""
    
    def __init__(self, 
                 azure_search_endpoint: str,
                 azure_search_key: str,
                 search_index_name: str = "electricity-documents"):
        """
        Initialize the match data tool
        
        Args:
            azure_search_endpoint: Azure Search endpoint
            azure_search_key: Azure Search admin key
            search_index_name: Name of the search index
        """
        
        self.search_client = SearchClient(
            endpoint=azure_search_endpoint,
            index_name=search_index_name,
            credential=AzureKeyCredential(azure_search_key)
        )
        
        print(f"✅ Match Data Tool initialized")
        print(f"   Azure Search: {azure_search_endpoint}")
        print(f"   Index: {search_index_name}")
    
    def load_excel_data(self, excel_path: str) -> pd.DataFrame:
        """
        Load data from EGA.xlsx file
        
        Args:
            excel_path: Path to the EGA.xlsx file
            
        Returns:
            DataFrame with Excel data
        """
        try:
            # Load Excel file
            df = pd.read_excel(excel_path)
            print(f"📊 Loaded Excel data: {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return pd.DataFrame()
    
    def search_vector_by_location(self, location: str) -> List[Dict[str, Any]]:
        """
        Search vector database for documents matching the location
        
        Args:
            location: Location to search for
            
        Returns:
            List of matching documents from vector database
        """
        try:
            # Search for documents with matching location
            search_results = self.search_client.search(
                search_text=f"{location}",
                filter=f"location eq '{location}' or search.ismatch('{location}', 'location')",
                select=["id", "filename", "content", "location", "client_name", 
                       "lease_start_date", "lease_end_date", "building_area", 
                       "area_unit", "building_type", "processed_at"],
                top=50
            )
            
            matches = []
            for result in search_results:
                matches.append({
                    "id": result.get("id"),
                    "filename": result.get("filename"),
                    "location": result.get("location"),
                    "client_name": result.get("client_name"),
                    "lease_start_date": result.get("lease_start_date"),
                    "lease_end_date": result.get("lease_end_date"),
                    "building_area": result.get("building_area"),
                    "area_unit": result.get("area_unit"),
                    "building_type": result.get("building_type"),
                    "processed_at": result.get("processed_at"),
                    "content_preview": (result.get("content", "")[:100] + "...") if result.get("content") else "",
                    "search_score": result.get("@search.score", 0)
                })
            
            return matches
            
        except Exception as e:
            print(f"❌ Error searching vector database for location '{location}': {e}")
            return []
    
    def fuzzy_search_vector_by_location(self, location: str) -> List[Dict[str, Any]]:
        """
        Perform fuzzy search for location in vector database
        
        Args:
            location: Location to search for
            
        Returns:
            List of matching documents from vector database
        """
        try:
            # Try multiple search strategies
            matches = []
            
            # Strategy 1: Exact search in location field
            try:
                exact_results = self.search_client.search(
                    search_text="*",
                    filter=f"location eq '{location}'",
                    select=["id", "filename", "content", "location", "client_name", 
                           "lease_start_date", "lease_end_date", "building_area", 
                           "area_unit", "building_type", "processed_at"],
                    top=20
                )
                matches.extend(list(exact_results))
            except:
                pass
            
            # Strategy 2: Text search across all fields
            try:
                text_results = self.search_client.search(
                    search_text=location,
                    select=["id", "filename", "content", "location", "client_name", 
                           "lease_start_date", "lease_end_date", "building_area", 
                           "area_unit", "building_type", "processed_at"],
                    top=20
                )
                matches.extend(list(text_results))
            except:
                pass
            
            # Strategy 3: Search for parts of the location (city, state, etc.)
            location_parts = location.replace(",", " ").split()
            for part in location_parts:
                if len(part) > 2:  # Skip very short words
                    try:
                        part_results = self.search_client.search(
                            search_text=part,
                            select=["id", "filename", "content", "location", "client_name", 
                                   "lease_start_date", "lease_end_date", "building_area", 
                                   "area_unit", "building_type", "processed_at"],
                            top=10
                        )
                        matches.extend(list(part_results))
                    except:
                        pass
            
            # Remove duplicates and format results
            unique_matches = {}
            for result in matches:
                doc_id = result.get("id")
                if doc_id and doc_id not in unique_matches:
                    unique_matches[doc_id] = {
                        "id": result.get("id"),
                        "filename": result.get("filename"),
                        "location": result.get("location"),
                        "client_name": result.get("client_name"),
                        "lease_start_date": result.get("lease_start_date"),
                        "lease_end_date": result.get("lease_end_date"),
                        "building_area": result.get("building_area"),
                        "area_unit": result.get("area_unit"),
                        "building_type": result.get("building_type"),
                        "processed_at": result.get("processed_at"),
                        "content_preview": (result.get("content", "")[:100] + "...") if result.get("content") else "",
                        "search_score": result.get("@search.score", 0)
                    }
            
            return list(unique_matches.values())
            
        except Exception as e:
            print(f"❌ Error performing fuzzy search for location '{location}': {e}")
            return []
    
    def match_excel_with_vector(self, excel_path: str, location_column: str = "location", 
                               column_2: str = None) -> List[Dict[str, Any]]:
        """
        Match Excel data with vector data based on location
        
        Args:
            excel_path: Path to EGA.xlsx file
            location_column: Name of location column in Excel (default: "location")
            column_2: Name of second column to include from Excel
            
        Returns:
            List of matched results showing both Excel and vector data
        """
        
        # Load Excel data
        excel_df = self.load_excel_data(excel_path)
        if excel_df.empty:
            return []
        
        # Auto-detect location column if not found
        if location_column not in excel_df.columns:
            # Try common location column names
            location_candidates = ["location", "Location", "LOCATION", "address", "Address", 
                                 "city", "City", "place", "Place"]
            for candidate in location_candidates:
                if candidate in excel_df.columns:
                    location_column = candidate
                    break
            else:
                print(f"⚠️  Location column '{location_column}' not found in Excel. Available columns: {list(excel_df.columns)}")
                # Use first column as location if no match found
                location_column = excel_df.columns[0]
                print(f"   Using '{location_column}' as location column")
        
        # Auto-detect column 2 if not specified
        if column_2 is None and len(excel_df.columns) > 1:
            column_2 = excel_df.columns[1]
        
        print(f"📍 Matching based on location column: '{location_column}'")
        if column_2:
            print(f"📋 Including column 2: '{column_2}'")
        
        matched_results = []
        
        for index, row in excel_df.iterrows():
            location = str(row[location_column]) if pd.notna(row[location_column]) else ""
            column_2_value = str(row[column_2]) if column_2 and pd.notna(row[column_2]) else ""
            
            if not location:
                print(f"⚠️  Row {index + 1}: No location data, skipping")
                continue
            
            print(f"\n🔍 Row {index + 1}: Searching for location '{location}'")
            
            # Search vector database
            vector_matches = self.fuzzy_search_vector_by_location(location)
            
            # Create result entry
            result_entry = {
                "excel_row": index + 1,
                "excel_data": {
                    "location": location,
                    "column_2": column_2_value if column_2 else "",
                    "all_columns": dict(row)
                },
                "vector_matches": vector_matches,
                "match_count": len(vector_matches)
            }
            
            matched_results.append(result_entry)
            
            if vector_matches:
                print(f"   ✅ Found {len(vector_matches)} vector matches")
            else:
                print(f"   ❌ No vector matches found")
        
        return matched_results
    
    def display_matches(self, matched_results: List[Dict[str, Any]]):
        """
        Display the matched results in a formatted way
        
        Args:
            matched_results: List of matched results from match_excel_with_vector
        """
        
        print(f"\n{'='*80}")
        print(f"📊 MATCH DATA RESULTS")
        print(f"{'='*80}")
        
        total_excel_rows = len(matched_results)
        total_matches = sum(result["match_count"] for result in matched_results)
        matched_rows = sum(1 for result in matched_results if result["match_count"] > 0)
        
        print(f"\n📈 SUMMARY:")
        print(f"   Excel rows processed: {total_excel_rows}")
        print(f"   Rows with vector matches: {matched_rows}")
        print(f"   Total vector matches found: {total_matches}")
        print(f"   Match rate: {(matched_rows/total_excel_rows*100):.1f}%")
        
        for result in matched_results:
            excel_row = result["excel_row"]
            excel_data = result["excel_data"]
            vector_matches = result["vector_matches"]
            
            print(f"\n{'-'*60}")
            print(f"📋 ROW {excel_row}")
            print(f"{'-'*60}")
            
            # Display Excel data
            print(f"📊 EXCEL DATA:")
            print(f"   Location: {excel_data['location']}")
            if excel_data.get("column_2"):
                print(f"   Column 2: {excel_data['column_2']}")
            
            # Show additional Excel columns if they exist
            other_columns = {k: v for k, v in excel_data["all_columns"].items() 
                           if k not in ["location", excel_data.get("column_2", "")]}
            if other_columns:
                print(f"   Other data: {dict(list(other_columns.items())[:3])}{'...' if len(other_columns) > 3 else ''}")
            
            # Display vector matches
            if vector_matches:
                print(f"\n🔍 VECTOR MATCHES ({len(vector_matches)}):")
                for i, match in enumerate(vector_matches[:5], 1):  # Show top 5 matches
                    print(f"\n   Match {i} (Score: {match['search_score']:.2f}):")
                    print(f"      File: {match['filename'] or 'Unknown'}")
                    print(f"      Location: {match['location'] or 'Not specified'}")
                    print(f"      Client: {match['client_name'] or 'Unknown'}")
                    if match['lease_start_date']:
                        print(f"      Lease Period: {match['lease_start_date']} to {match.get('lease_end_date', 'Unknown')}")
                    if match['building_area']:
                        print(f"      Area: {match['building_area']} {match.get('area_unit', '')}")
                    if match['building_type']:
                        print(f"      Type: {match['building_type']}")
                
                if len(vector_matches) > 5:
                    print(f"   ... and {len(vector_matches) - 5} more matches")
            else:
                print(f"\n❌ NO VECTOR MATCHES FOUND")
    
    def export_matches_to_excel(self, matched_results: List[Dict[str, Any]], 
                               output_path: str = "match_results.xlsx"):
        """
        Export matched results to Excel file
        
        Args:
            matched_results: List of matched results
            output_path: Path for output Excel file
        """
        try:
            # Flatten the data for Excel export
            export_data = []
            
            for result in matched_results:
                excel_row = result["excel_row"]
                excel_data = result["excel_data"]
                vector_matches = result["vector_matches"]
                
                if vector_matches:
                    # Create a row for each vector match
                    for i, match in enumerate(vector_matches):
                        export_row = {
                            "Excel_Row": excel_row,
                            "Excel_Location": excel_data["location"],
                            "Excel_Column_2": excel_data.get("column_2", ""),
                            "Match_Number": i + 1,
                            "Vector_Filename": match["filename"],
                            "Vector_Location": match["location"],
                            "Vector_Client": match["client_name"],
                            "Vector_Lease_Start": match["lease_start_date"],
                            "Vector_Lease_End": match["lease_end_date"],
                            "Vector_Building_Area": match["building_area"],
                            "Vector_Area_Unit": match["area_unit"],
                            "Vector_Building_Type": match["building_type"],
                            "Search_Score": match["search_score"]
                        }
                        export_data.append(export_row)
                else:
                    # Create a row even if no matches found
                    export_row = {
                        "Excel_Row": excel_row,
                        "Excel_Location": excel_data["location"],
                        "Excel_Column_2": excel_data.get("column_2", ""),
                        "Match_Number": 0,
                        "Vector_Filename": "NO MATCH",
                        "Vector_Location": "",
                        "Vector_Client": "",
                        "Vector_Lease_Start": "",
                        "Vector_Lease_End": "",
                        "Vector_Building_Area": "",
                        "Vector_Area_Unit": "",
                        "Vector_Building_Type": "",
                        "Search_Score": 0
                    }
                    export_data.append(export_row)
            
            # Create DataFrame and export
            export_df = pd.DataFrame(export_data)
            export_df.to_excel(output_path, index=False)
            
            print(f"\n💾 Results exported to: {output_path}")
            print(f"   Rows exported: {len(export_data)}")
            
        except Exception as e:
            print(f"❌ Error exporting to Excel: {e}")


def main():
    """Main function to run the match data tool"""
    
    # Get configuration from environment variables
    azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
    azure_search_key = os.getenv('AZURE_SEARCH_KEY')
    search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'electricity-documents')
    
    if not azure_search_endpoint or not azure_search_key:
        print("❌ Missing required environment variables:")
        print("   AZURE_SEARCH_ENDPOINT")
        print("   AZURE_SEARCH_KEY")
        print("   Optional: AZURE_SEARCH_INDEX_NAME (defaults to 'electricity-documents')")
        return
    
    # Initialize the tool
    try:
        tool = MatchDataTool(
            azure_search_endpoint=azure_search_endpoint,
            azure_search_key=azure_search_key,
            search_index_name=search_index_name
        )
    except Exception as e:
        print(f"❌ Failed to initialize Match Data Tool: {e}")
        return
    
    # Check for EGA.xlsx file
    excel_file = "EGA.xlsx"
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        print(f"   Please ensure EGA.xlsx is in the current directory")
        return
    
    print(f"\n🚀 Starting match data process...")
    
    # Perform matching
    matched_results = tool.match_excel_with_vector(excel_file)
    
    if matched_results:
        # Display results
        tool.display_matches(matched_results)
        
        # Export results
        tool.export_matches_to_excel(matched_results)
        
        print(f"\n✅ Match data process completed successfully!")
    else:
        print(f"\n❌ No results to display")


if __name__ == "__main__":
    main()