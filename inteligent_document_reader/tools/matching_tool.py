"""
Matching Tool for matching Excel data with vector database data based on location
"""

import json
from typing import Any
from langchain.tools import BaseTool
from pydantic import Field


class MatchingTool(BaseTool):
    """Tool for matching Excel data with vector database entries based on location"""
    
    name: str = "excel_matching_tool"
    description: str = "Matches data from Excel file with vector database entries (leases, electricity, and natural gas) based on location. Input should be JSON with 'excel_path' and optional 'location_column'."
    search_client: Any = Field(default=None, exclude=True)
    utilities_search_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, search_client, utilities_search_client=None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'search_client', search_client)
        object.__setattr__(self, 'utilities_search_client', utilities_search_client or search_client)
    
    def _fuzzy_search_vector_by_location(self, location: str):
        """Search vector database using same logic as original MatchDataTool"""
        try:
            all_results = []
            
            # STRATEGY 1: Search LEASE index
            try:
                print(f"   🔍 Searching lease index for: {location}")
                
                lease_search_results = self.search_client.search(
                    search_text=f"{location}",
                    select=["id", "filename", "location", "client_name", "lease_start_date", 
                           "lease_end_date", "building_area", "area_unit", "building_type", "processed_at"],
                    top=100
                )
                
                # Also try filter-based search for exact client matches
                try:
                    lease_filtered_results = self.search_client.search(
                        search_text="*",
                        filter=f"client_name eq '{location}'",
                        select=["id", "filename", "location", "client_name", "lease_start_date", 
                               "lease_end_date", "building_area", "area_unit", "building_type", "processed_at"],
                        top=100
                    )
                    lease_results = list(lease_search_results) + list(lease_filtered_results)
                except:
                    lease_results = list(lease_search_results)
                
                all_results.extend(lease_results)
                print(f"   ✅ Found {len(lease_results)} results in lease index")
            except Exception as e:
                print(f"   ⚠️ Lease index search failed: {e}")
            
            # STRATEGY 2: Search UTILITIES index (if different client)
            if self.utilities_search_client != self.search_client:
                try:
                    print(f"   🔍 Searching utilities index for: {location}")
                    utilities_start_count = len(all_results)
                    
                    utilities_search_results = self.utilities_search_client.search(
                        search_text=f"{location}",
                        select=["id", "filename", "content", "location", "client_name", "doc_type",
                               "vendor_name", "invoice_date", "measurement_period_start",
                               "measurement_period_end", "consumption_amount", "unit_of_measure", 
                               "processed_at", "page_no"],
                        top=100
                    )
                    all_results.extend(list(utilities_search_results))
                    
                    try:
                        utilities_filtered_results = self.utilities_search_client.search(
                            search_text="*",
                            filter=f"client_name eq '{location}'",
                            select=["id", "filename", "content", "location", "client_name", "doc_type",
                                   "vendor_name", "invoice_date", "measurement_period_start",
                                   "measurement_period_end", "consumption_amount", "unit_of_measure", 
                                   "processed_at", "page_no"],
                            top=100
                        )
                        all_results.extend(list(utilities_filtered_results))
                    except:
                        pass
                        
                    utilities_count = len(all_results) - utilities_start_count
                    print(f"   ✅ Found {utilities_count} results in utilities index")
                except Exception as e:
                    print(f"   ⚠️ Utilities index search failed: {e}")
            
            # Process results same as original
            matches = []
            for result in all_results:
                doc_type = result.get("doc_type")
                
                if doc_type in ["Electricity", "NaturalGas"]:
                    match_data = {
                        "id": result.get("id"),
                        "filename": result.get("filename"),
                        "location": result.get("location"),
                        "client_name": result.get("client_name"),
                        "doc_type": doc_type,
                        "vendor_name": result.get("vendor_name"),
                        "invoice_date": result.get("invoice_date"),
                        "measurement_period_start": result.get("measurement_period_start"),
                        "measurement_period_end": result.get("measurement_period_end"),
                        "consumption_amount": result.get("consumption_amount"),
                        "unit_of_measure": result.get("unit_of_measure"),
                        "search_score": result.get("@search.score", 0)
                    }
                else:
                    match_data = {
                        "id": result.get("id"),
                        "filename": result.get("filename"),
                        "location": result.get("location"),
                        "client_name": result.get("client_name"),
                        "lease_start_date": result.get("lease_start_date"),
                        "lease_end_date": result.get("lease_end_date"),
                        "building_area": result.get("building_area"),
                        "area_unit": result.get("area_unit"),
                        "building_type": result.get("building_type"),
                        "search_score": result.get("@search.score", 0),
                        "doc_type": "Lease"
                    }
                
                matches.append(match_data)
            
            # Group by filename to avoid counting pages as separate documents
            documents = {}
            for match in matches:
                filename = match.get("filename", "Unknown")
                if filename not in documents:
                    documents[filename] = match
            
            return list(documents.values())
            
        except Exception as e:
            print(f"   ❌ Error in location search: {e}")
            return []
    
    def _run(self, input_data: str) -> str:
        """Match Excel data with vector database data by searching column 2 locations"""
        try:
            data = json.loads(input_data)
            excel_path = data.get('excel_path', 'EGA.xlsx')
            
            # Load Excel data
            import pandas as pd
            try:
                excel_df = pd.read_excel(excel_path)
            except Exception as e:
                return f"❌ Error loading Excel file '{excel_path}': {e}"
            
            if excel_df.empty:
                return f"❌ Excel file '{excel_path}' is empty"
            
            if len(excel_df.columns) < 2:
                return f"❌ Excel file needs at least 2 columns. Found: {list(excel_df.columns)}"
            
            # Use column 2 (index 1) as the location to search for
            column_2 = excel_df.columns[1]
            
            print(f"   📊 Processing Excel file: {excel_path}")
            print(f"   🔍 Searching using column 2: {column_2}")
            
            results = []
            total_matches = 0
            matched_rows = 0
            
            for index, row in excel_df.iterrows():
                row_data = dict(row)
                search_location = str(row[column_2]) if pd.notna(row[column_2]) else ""
                
                if not search_location.strip():
                    results.append({
                        "excel_row": index + 1,
                        "search_location": search_location,
                        "excel_data": row_data,
                        "vector_matches": 0,
                        "matches": []
                    })
                    continue
                
                print(f"   🔍 Row {index + 1}: Searching for '{search_location}'")
                
                vector_matches = self._fuzzy_search_vector_by_location(search_location)
                
                if vector_matches:
                    matched_rows += 1
                    total_matches += len(vector_matches)
                    print(f"      ✅ Found {len(vector_matches)} matches")
                else:
                    print(f"      ❌ No matches found")
                
                results.append({
                    "excel_row": index + 1,
                    "search_location": search_location,
                    "excel_data": row_data,
                    "vector_matches": len(vector_matches),
                    "matches": vector_matches[:5]  # Top 5 matches for display
                })
            
            # Format response
            response = f"📊 EXCEL MATCH RESULTS\n"
            response += f"{'='*50}\n\n"
            response += f"📈 SUMMARY:\n"
            response += f"   Excel rows processed: {len(results)}\n"
            response += f"   Rows with vector matches: {matched_rows}\n"
            response += f"   Total vector matches: {total_matches}\n"
            response += f"   Match rate: {(matched_rows/len(results)*100):.1f}%\n\n"
            
            response += f"📋 DETAILED RESULTS:\n"
            for result in results[:10]:  # Show first 10 results
                response += f"\n--- Row {result['excel_row']} ---\n"
                response += f"🔍 Searched for: '{result['search_location']}'\n"
                response += f"📊 Excel Data: {dict(list(result['excel_data'].items())[:3])}{'...' if len(result['excel_data']) > 3 else ''}\n"
                
                if result['matches']:
                    response += f"✅ Vector Matches ({result['vector_matches']}):\n"
                    for i, match in enumerate(result['matches'], 1):
                        doc_type = match.get('doc_type', 'Lease')
                        response += f"  {i}. {match['filename']} ({doc_type}) (Score: {match['search_score']:.2f})\n"
                        response += f"     Location: {match['location']}\n"
                        response += f"     Client: {match['client_name']}\n"
                        
                        if doc_type == 'Lease':
                            if match.get('lease_start_date'):
                                response += f"     Lease: {match['lease_start_date']} - {match.get('lease_end_date')}\n"
                            if match.get('building_area'):
                                response += f"     Area: {match['building_area']} {match.get('area_unit', '')}\n"
                        elif doc_type in ['Electricity', 'NaturalGas']:
                            if match.get('vendor_name'):
                                response += f"     Vendor: {match['vendor_name']}\n"
                            if match.get('consumption_amount'):
                                response += f"     Consumption: {match['consumption_amount']} {match.get('unit_of_measure', '')}\n"
                else:
                    response += f"❌ No vector matches found\n"
            
            if len(results) > 10:
                response += f"\n... and {len(results) - 10} more rows\n"
            
            return response
            
        except json.JSONDecodeError:
            return "❌ Error: Invalid JSON input format. Use: {'excel_path': 'EGA.xlsx'}"
        except Exception as e:
            return f"❌ Error matching Excel data: {str(e)}"