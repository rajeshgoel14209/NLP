Akshat - continue on tool selection accuracy and will provide the results for given set of questions.

Suhani - Create a streamlit app - with below features -

         ################################################################################################################################
		 
		                                         MODEL EVALUATION DASHBOARD IN STREAMLIT
		 
		 ################################################################################################################################



         ################################
		 
		        UPLOAD TEST CASES 
		 
		 ################################

         Step 01 - Upload the test cases file (start testing with 8 -10 questions) ==>  should have questions , answers , retriever chunk , retriever chunk id (optional)
		 
		 
		 
         ################################
		 
		        START RETREIVER ENGINE
		 
		 ################################		 
		 
		 step 02 - Start retriever engine on clicking retriever engine button.
		 
		           Step 01 - Set the retriver count on UI and Hit retriever API and save all the top k chunks in the sheet ( add each chunk in new column of sheet for give question )
				   Step 02 - Once completed , enable downoad button to download test cases file updated with retriever results.
				   
		 step 03 - Click on download button to download retriever file.
		 
         ################################
		 
		        START LLM ENGINE
		 
		 ################################	

         step 04 - Upload test cases with retriever results.		 
				   
		 step 05 - Start LLM engine on clicking retriever engine button.
		 
		           Step 01 - Hit generate answer API and generate answer using retriever context and question and save response in the sheet
				   Step 02 - Once completed , enable downoad button to download test cases file updated with retriever results and LLM response.
				   Step 03 - Calculate acuracy also				   


         ################################
		 
		        START EVALUATION ENGINE
		 
		 ################################
		 
		 step 06 - Select evaluation parameter from drop down on UI
		 step 07 - Click on evaluate LLM response and generate score
