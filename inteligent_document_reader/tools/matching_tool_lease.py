"""
Lease Matching Tool for Excel validation - reads input Excel and writes output Excel with lease matches
"""

import json
import pandas as pd
from pathlib import Path
from typing import Any
from langchain.tools import BaseTool
from pydantic import Field


class MatchingToolLease(BaseTool):
    """Tool for matching Excel data with lease documents and outputting to Excel format"""
    
    name: str = "excel_lease_matching_tool"
    description: str = "Reads input Excel file (rows 6+, columns B-J), matches locations with lease database, writes results to output_lease.xlsm (columns K-Q). Input should be JSON with 'input_excel_path'."
    search_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, search_client, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'search_client', search_client)
    
    def _search_lease_by_location(self, location: str):
        """Search lease database for location matches"""
        try:
            print(f"   🔍 Searching lease database for: {location}")
            
            # Search for lease documents with matching location
            search_results = self.search_client.search(
                search_text=f"{location}",
                filter="doc_type eq 'Lease'",  # Only lease documents
                select=["id", "filename", "location", "client_name", "lease_start_date", 
                       "lease_end_date", "building_area", "area_unit", "building_type", "processed_at"],
                top=50
            )
            
            # Also try filter-based search for exact client matches
            try:
                filtered_results = self.search_client.search(
                    search_text="*",
                    filter=f"doc_type eq 'Lease' and client_name eq '{location}'",
                    select=["id", "filename", "location", "client_name", "lease_start_date", 
                           "lease_end_date", "building_area", "area_unit", "building_type", "processed_at"],
                    top=50
                )
                all_results = list(search_results) + list(filtered_results)
            except:
                all_results = list(search_results)
            
            # Process results and find best match
            matches = []
            for result in all_results:
                matches.append({
                    "filename": result.get("filename"),
                    "location": result.get("location"),
                    "client_name": result.get("client_name"),
                    "lease_start_date": result.get("lease_start_date"),
                    "lease_end_date": result.get("lease_end_date"),
                    "building_area": result.get("building_area"),
                    "area_unit": result.get("area_unit"),
                    "building_type": result.get("building_type"),
                    "score": result.get("@search.score", 0)
                })
            
            # Group by filename to avoid counting pages as separate documents
            documents = {}
            for match in matches:
                filename = match.get("filename", "Unknown")
                if filename not in documents:
                    documents[filename] = match  # Take first page data for the document
            
            unique_matches = list(documents.values())
            
            # Return best match (highest score)
            if unique_matches:
                best_match = max(unique_matches, key=lambda x: x.get("score", 0))
                print(f"      ✅ Found {len(unique_matches)} matches, best: {best_match['filename']}")
                return best_match
            else:
                print(f"      ❌ No matches found")
                return None
            
        except Exception as e:
            print(f"   ❌ Error searching for location: {e}")
            return None
    
    def _run(self, input_data: str) -> str:
        """Process Excel file and create output with lease matches"""
        try:
            data = json.loads(input_data)
            input_excel_path = data.get('input_excel_path', 'Leases_example.xlsm')
            output_excel_path = data.get('output_excel_path', 'output_lease.xlsm')
            
            print(f"📊 LEASE MATCHING TOOL")
            print(f"   Input: {input_excel_path}")
            print(f"   Output: {output_excel_path}")
            
            # Read the input Excel file
            try:
                # Read with no header to access raw structure
                df_raw = pd.read_excel(input_excel_path, header=None, engine='openpyxl')
                print(f"   📈 Loaded Excel file: {df_raw.shape}")
            except Exception as e:
                return f"❌ Error loading Excel file '{input_excel_path}': {e}"
            
            # Copy the entire structure to output
            df_output = df_raw.copy()
            
            # Find data rows (starting from row 6, which is index 5)
            data_start_row = 5  # Row 6 (0-indexed as 5)
            
            if len(df_output) <= data_start_row:
                return f"❌ Excel file has no data rows. Expected data starting from row 6."
            
            processed_rows = 0
            matched_rows = 0
            
            # Process each data row
            for row_idx in range(data_start_row, len(df_output)):
                # Read input data from columns B-J (indices 1-9)
                location = df_output.iloc[row_idx, 1]  # Column B
                building_type = df_output.iloc[row_idx, 2]  # Column C
                emission_source = df_output.iloc[row_idx, 3]  # Column D
                measurement_date = df_output.iloc[row_idx, 4]  # Column E
                building_area = df_output.iloc[row_idx, 5]  # Column F
                area_unit = df_output.iloc[row_idx, 6]  # Column G
                consumption_unit = df_output.iloc[row_idx, 7]  # Column H
                annual_consumption = df_output.iloc[row_idx, 8]  # Column I
                lease_details = df_output.iloc[row_idx, 9]  # Column J
                
                # Skip rows with no location data
                if pd.isna(location) or str(location).strip() == '':
                    continue
                
                processed_rows += 1
                print(f"\n📍 Row {row_idx + 1}: Processing location '{location}'")
                
                # Search for lease matches
                lease_match = self._search_lease_by_location(str(location))
                
                if lease_match:
                    matched_rows += 1
                    
                    # Write lease data to columns K-Q (indices 10-16)
                    df_output.iloc[row_idx, 10] = lease_match.get('location', '')  # Column K
                    df_output.iloc[row_idx, 11] = lease_match.get('lease_start_date', '')  # Column L
                    df_output.iloc[row_idx, 12] = lease_match.get('lease_end_date', '')  # Column M
                    df_output.iloc[row_idx, 13] = lease_match.get('building_area', '')  # Column N
                    df_output.iloc[row_idx, 14] = lease_match.get('area_unit', '')  # Column O
                    df_output.iloc[row_idx, 15] = lease_match.get('building_type', '')  # Column P
                    df_output.iloc[row_idx, 16] = lease_match.get('filename', '')  # Column Q - source file
                    
                    print(f"   ✅ Matched with lease: {lease_match.get('filename', 'Unknown')}")
                    print(f"      Location: {lease_match.get('location', 'N/A')}")
                    print(f"      Lease: {lease_match.get('lease_start_date', 'N/A')} to {lease_match.get('lease_end_date', 'N/A')}")
                    print(f"      Area: {lease_match.get('building_area', 'N/A')} {lease_match.get('area_unit', '')}")
                else:
                    # Clear columns K-Q if no match
                    for col_idx in range(10, 17):
                        df_output.iloc[row_idx, col_idx] = ''
                    print(f"   ❌ No lease match found")
            
            # Save the output Excel file
            try:
                df_output.to_excel(output_excel_path, header=False, index=False, engine='openpyxl')
                print(f"\n💾 Saved output to: {output_excel_path}")
            except Exception as e:
                return f"❌ Error saving output file: {e}"
            
            # Summary
            result = f"📊 LEASE MATCHING COMPLETED\n"
            result += f"{'='*40}\n"
            result += f"📁 Input file: {input_excel_path}\n"
            result += f"📁 Output file: {output_excel_path}\n"
            result += f"📈 Summary:\n"
            result += f"   • Rows processed: {processed_rows}\n"
            result += f"   • Rows with lease matches: {matched_rows}\n"
            result += f"   • Match rate: {(matched_rows/processed_rows*100):.1f}% \n" if processed_rows > 0 else "   • Match rate: 0%\n"
            result += f"   • Output columns: K-Q populated with lease data\n"
            result += f"\n✅ Results written to {output_excel_path}"
            
            return result
            
        except json.JSONDecodeError:
            return "❌ Error: Invalid JSON input format. Use: {'input_excel_path': 'Leases_example.xlsm'}"
        except Exception as e:
            return f"❌ Error processing Excel matching: {str(e)}"