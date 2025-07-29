#!/usr/bin/env python3
"""
End-to-end Integration Script
Processes all lease and utilities documents and runs match excel functionality
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import traceback

# Import the lease agent and required modules
from lease_agent import create_lease_agent_from_config
from config import get_config


class End2EndIntegration:
    """End-to-end integration for processing all documents and matching with Excel"""
    
    def __init__(self):
        """Initialize the integration with the lease agent"""
        try:
            print("🚀 Initializing End-to-End Integration...")
            
            # Load configuration
            config = get_config()
            print(f"   📋 Configuration loaded")
            
            # Create lease agent instance
            self.agent = create_lease_agent_from_config(config)
            print(f"   ✅ Lease agent initialized successfully")
            
        except Exception as e:
            print(f"   ❌ Failed to initialize integration: {e}")
            raise
    
    def ingest_all_documents(self) -> Dict[str, Any]:
        """
        Ingest all lease and utilities documents from their respective folders
        
        Returns:
            Dictionary with results from all ingestion operations
        """
        print("\n📥 STARTING DOCUMENT INGESTION")
        print("="*50)
        
        results = {
            "lease_ingestion": None,
            "electric_ingestion": None,
            "natural_gas_ingestion": None,
            "overall_status": "pending"
        }
        
        try:
            # 1. Ingest lease documents from /leases folder
            print("\n📁 Processing lease documents...")
            lease_result = self.agent.ask_question("ingest all lease documents")
            results["lease_ingestion"] = lease_result
            
            if lease_result["status"] == "success":
                print("   ✅ Lease documents processed successfully")
            else:
                print(f"   ⚠️ Lease ingestion issue: {lease_result.get('error', 'Unknown error')}")
            
            # 2. Ingest electric documents from /electric folder  
            print("\n⚡ Processing electric documents...")
            electric_result = self.agent.ask_question("process all electric bills")
            results["electric_ingestion"] = electric_result
            
            if electric_result["status"] == "success":
                print("   ✅ Electric documents processed successfully")
            else:
                print(f"   ⚠️ Electric ingestion issue: {electric_result.get('error', 'Unknown error')}")
            
            # 3. Ingest natural gas documents from /natural_gas folder
            print("\n🔥 Processing natural gas documents...")
            gas_result = self.agent.ask_question("ingest natural gas documents")
            results["natural_gas_ingestion"] = gas_result
            
            if gas_result["status"] == "success":
                print("   ✅ Natural gas documents processed successfully")
            else:
                print(f"   ⚠️ Natural gas ingestion issue: {gas_result.get('error', 'Unknown error')}")
            
            # Determine overall status
            successful_ingestions = sum(1 for r in [lease_result, electric_result, gas_result] 
                                      if r and r["status"] == "success")
            
            if successful_ingestions == 3:
                results["overall_status"] = "all_successful"
                print("\n🎉 ALL DOCUMENT INGESTION COMPLETED SUCCESSFULLY!")
            elif successful_ingestions > 0:
                results["overall_status"] = "partial_success"
                print(f"\n⚠️ PARTIAL SUCCESS: {successful_ingestions}/3 document types processed successfully")
            else:
                results["overall_status"] = "failed"
                print("\n❌ ALL DOCUMENT INGESTION FAILED")
                
        except Exception as e:
            print(f"\n❌ Error during document ingestion: {e}")
            results["overall_status"] = "error"
            results["error"] = str(e)
            
        return results
    
    def run_match_excel(self, excel_path: str = "EGA.xlsx") -> Dict[str, Any]:
        """
        Run the match excel functionality to correlate Excel data with vector database
        
        Args:
            excel_path: Path to the Excel file (default: EGA.xlsx)
            
        Returns:
            Dictionary with match results
        """
        print(f"\n📊 STARTING EXCEL MATCHING")
        print("="*50)
        
        try:
            # Check if Excel file exists
            if not Path(excel_path).exists():
                return {
                    "status": "error",
                    "error": f"Excel file not found: {excel_path}",
                    "excel_path": excel_path
                }
            
            print(f"   📋 Excel file found: {excel_path}")
            
            # Use the match_data tool through the agent
            query = f"""
            Use the match_data tool to match the Excel file '{excel_path}' with all available vector database data (leases, electricity, and natural gas documents). 
            Search using column 2 from the Excel file and find matching documents in the vector database based on location.
            Provide a comprehensive summary of matches found.
            """
            
            print("   🔍 Running match analysis...")
            result = self.agent.ask_question(query)
            
            if result["status"] == "success":
                print("   ✅ Excel matching completed successfully")
                return {
                    "status": "success",
                    "excel_path": excel_path,
                    "result": result["result"]
                }
            else:
                print(f"   ❌ Excel matching failed: {result.get('error', 'Unknown error')}")
                return {
                    "status": "error",
                    "excel_path": excel_path,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            print(f"   ❌ Exception during Excel matching: {e}")
            return {
                "status": "error",
                "excel_path": excel_path,
                "error": str(e)
            }
    
    def run_full_integration(self, excel_path: str = "EGA.xlsx") -> Dict[str, Any]:
        """
        Run the complete end-to-end integration:
        1. Ingest all lease documents
        2. Ingest all utilities documents  
        3. Run Excel matching
        
        Args:
            excel_path: Path to the Excel file for matching (default: EGA.xlsx)
            
        Returns:
            Dictionary with complete results
        """
        print("🎯 STARTING FULL END-TO-END INTEGRATION")
        print("="*60)
        
        integration_results = {
            "ingestion_results": None,
            "match_results": None,
            "overall_status": "running",
            "excel_path": excel_path
        }
        
        try:
            # Step 1: Ingest all documents
            print("\n🔄 STEP 1: Document Ingestion")
            ingestion_results = self.ingest_all_documents()
            integration_results["ingestion_results"] = ingestion_results
            
            # Step 2: Run Excel matching (regardless of ingestion status to check existing data)
            print("\n🔄 STEP 2: Excel Matching")
            match_results = self.run_match_excel(excel_path)
            integration_results["match_results"] = match_results
            
            # Determine overall status
            ingestion_status = ingestion_results.get("overall_status", "failed")
            match_status = match_results.get("status", "failed")
            
            if ingestion_status in ["all_successful", "partial_success"] and match_status == "success":
                integration_results["overall_status"] = "success"
                print("\n🎉 END-TO-END INTEGRATION COMPLETED SUCCESSFULLY!")
                print("   ✅ All documents processed and Excel matching completed")
            elif ingestion_status in ["all_successful", "partial_success"]:
                integration_results["overall_status"] = "partial_success"
                print("\n⚠️ INTEGRATION PARTIALLY SUCCESSFUL")
                print("   ✅ Documents processed but Excel matching had issues")
            elif match_status == "success":
                integration_results["overall_status"] = "partial_success"
                print("\n⚠️ INTEGRATION PARTIALLY SUCCESSFUL")
                print("   ✅ Excel matching worked but document ingestion had issues")
            else:
                integration_results["overall_status"] = "failed"
                print("\n❌ INTEGRATION FAILED")
                print("   ❌ Both document ingestion and Excel matching had issues")
                
        except Exception as e:
            print(f"\n❌ Critical error during integration: {e}")
            integration_results["overall_status"] = "error"
            integration_results["error"] = str(e)
            integration_results["traceback"] = traceback.format_exc()
        
        return integration_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of the integration results"""
        print("\n" + "="*60)
        print("📋 END-TO-END INTEGRATION SUMMARY")
        print("="*60)
        
        overall_status = results.get("overall_status", "unknown")
        
        # Overall Status
        status_icons = {
            "success": "🎉",
            "partial_success": "⚠️",
            "failed": "❌",
            "error": "💥"
        }
        
        icon = status_icons.get(overall_status, "❓")
        print(f"\n{icon} OVERALL STATUS: {overall_status.upper().replace('_', ' ')}")
        
        # Ingestion Summary
        ingestion = results.get("ingestion_results", {})
        if ingestion:
            print(f"\n📥 DOCUMENT INGESTION:")
            
            lease_status = "✅" if ingestion.get("lease_ingestion", {}).get("status") == "success" else "❌"
            electric_status = "✅" if ingestion.get("electric_ingestion", {}).get("status") == "success" else "❌"
            gas_status = "✅" if ingestion.get("natural_gas_ingestion", {}).get("status") == "success" else "❌"
            
            print(f"   {lease_status} Lease documents")
            print(f"   {electric_status} Electric documents") 
            print(f"   {gas_status} Natural gas documents")
        
        # Match Summary
        match_results = results.get("match_results", {})
        if match_results:
            print(f"\n📊 EXCEL MATCHING:")
            match_status = "✅" if match_results.get("status") == "success" else "❌"
            excel_path = match_results.get("excel_path", "Unknown")
            print(f"   {match_status} Excel file: {excel_path}")
        
        # Errors
        if results.get("error"):
            print(f"\n❌ CRITICAL ERROR: {results['error']}")
        
        print("\n" + "="*60)


def main():
    """Main function to run the end-to-end integration"""
    try:
        # Create integration instance
        integration = End2EndIntegration()
        
        # Determine Excel file path
        excel_path = "EGA.xlsx"
        if len(sys.argv) > 1:
            excel_path = sys.argv[1]
        
        # Run full integration
        results = integration.run_full_integration(excel_path)
        
        # Print comprehensive summary
        integration.print_summary(results)
        
        # Exit with appropriate code
        if results["overall_status"] == "success":
            sys.exit(0)
        elif results["overall_status"] == "partial_success":
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(3)


if __name__ == "__main__":
    main()