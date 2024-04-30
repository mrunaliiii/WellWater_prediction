import streamlit as st
import pandas as pd
import pickle

# Load the machine learning model
model_file_path = './Model/random_forest_model.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

dataset_path = './Data/Current_Draft.csv'
df = pd.read_csv(dataset_path)

# Load water level data
water_level_path = './Data/well water level.csv'
water_level_df = pd.read_csv(water_level_path)

# Streamlit app title
st.title("Current Water Level with NAQUIM data")
districts = {
        "ANDHRA PRADESH":['Anantapur','Chittoor','East Godavari','Guntur','Kadapa','Krishna','Kurnool','Nellore','Prakasam','Srikakulam','Visakhapatnam','Vizianagaram','West Godavari'],
        "ARUNACHAL PRADESH":['Anjaw','Changlang','Dibang Valley','East Kameng','East Siang','Kurung Kumey','Lohit','Lower Dibang Valley','Lower Subansiri','Papum Pare','Tawang','Tirap','Upper Siang','Upper Subansiri','West Kameng','West Siang'],
        "ASSAM":['Baksa', 'Barpeta', 'Bongaigaon', 'Cachar', 'Chirang', 'Darrang', 'Dhemaji', 'Dhubri', 'Dibrugarh', 'Dima Hasao', 'Goalpara', 'Golaghat', 'Hailakandi', 'Jorhat', 'Kamrup', 'Kamrup Metro (Rural)', 'Kamrup Metro (U)', 'Karbi Anglong', 'Karimganj', 'Kokrajhar', 'Lakhimpur', 'Morigaon', 'Nagaon', 'Nalbari', 'Sivasagar', 'Sonitpur', 'Tinsukia','Udalguri'],
        "BIHAR":['Araria', 'Arwal', 'Aurangabad', 'Banka', 'Begusarai', 'Bhabhua', 'Bhagalpur', 'Bhojpur', 'Buxar', 'Darbhanga', 'East Champaran', 'Gaya', 'Gopalganj', 'Jamui', 'Jehanabad', 'Katihar', 'Khagaria', 'Kishanganj', 'Lakhisarai', 'Madhepura', 'Madhubani', 'Munger', 'Muzaffarpur', 'Nalanda', 'Nawada', 'Patna', 'Purnia', 'Rohtas', 'Saharsa', 'Samastipur', 'Saran', 'Sheikhpura', 'Sheohar', 'Sitamarhi', 'Siwan', 'Supaul', 'Vaishali', 'West Champaran'],
        "CHATTISGARH": ['Balod', 'Baloda Bazar', 'Balrampur', 'Bastar', 'Bemetara', 'Bijapur', 'Bilaspur', 'Dantewara', 'Dhamtari', 'Durg', 'Gariaband', 'Janjgir-Champa', 'Jashpur', 'Kanker', 'Kawardha', 'Kondagaon', 'Korba', 'Koriya', 'Mahasamund', 'Mungeli', 'Narayanpur', 'Raigarh', 'Raipur', 'Rajnandgaon', 'Sukma', 'Surajpur', 'Surguja'],
        "DELHI": ['Central Delhi', 'East Delhi', 'New Delhi', 'North Delhi', 'North East Delhi', 'North West Delhi', 'Shahdara', 'South Delhi', 'South East Delhi', 'South West Delhi', 'West Delhi', 'Non Revenue Unit'],
        "GOA": ['North Goa','South Goa'],
        "GUJARAT": ['Ahmedabad', 'Amreli', 'Anand', 'Aravalli', 'Banaskantha', 'Bharuch', 'Bhavnagar', 'Botad', 'Chhota Udaipur', 'Dahod', 'Dang', 'Devbhumi Dwarka', 'Gandhinagar', 'Gir Somnath', 'Jamnagar', 'Junagadh', 'Kachchh', 'Kheda', 'Mahesana', 'Mahisagar', 'Morbi', 'Narmada', 'Navsari', 'Panchmahal', 'Patan', 'Porbandar', 'Rajkot', 'Sabarkantha', 'Surat', 'Surendranagar', 'Tapi', 'Vadodara', 'Valsad'],
        "HARYANA": ['Ambala', 'Bhiwani', 'Charki Dadri', 'Faridabad', 'Fatehabad', 'Gurugram', 'Hisar', 'Jhajjar', 'Jind', 'Kaithal', 'Karnal', 'Kurukshetra', 'Mahendragarh', 'Mewat', 'Palwal', 'Panchkula', 'Panipat', 'Rewari', 'Rohtak', 'Sirsa', 'Sonipat', 'Yamunanagar'],
        "HIMACHAL PRADESH": ['Indora', 'Nurpur', 'Balh', 'Paonta', 'Kala Amb', 'Nalagarh', 'Una', 'Hum'],
        "J&K": ['Anantnag', 'Bandipora', 'Baramulla', 'Budgam', 'Doda', 'Ganderbal', 'Jammu', 'Kathua', 'Kishtwar', 'Kulgam', 'Kupwara', 'Leh', 'Kargil', 'Pulwama', 'Rajouri', 'Poonch', 'Ramban', 'Reasi', 'Samba', 'Shopian', 'Srinagar', 'Udhampur'],
        "JHARKHAND": ['Bokaro', 'Chatra', 'Deoghar', 'Dhanbad', 'Dumka', 'East Singhbhum', 'Garhwa', 'Giridih', 'Godda', 'Gumla', 'Hazaribagh', 'Jamtara', 'Khunti', 'Koderma', 'Latehar', 'Lohardaga', 'Pakur', 'Palamau', 'Ramgarh', 'Ranchi', 'Sahebganj', 'Saraikela - Kharsawan', 'Simdega', 'West Singhbhum'],
        "KARNATAKA": ['Bagalkote', 'Ballari', 'Belagavi', 'Bengaluru Rural', 'Bengaluru Urban', 'Bidar', 'Chamrajnagara', 'Chikballapur', 'Chikkamagaluru', 'Chitradurga', 'Dakshin Kannada', 'Davangere', 'Dharwad', 'Gadag', 'Hassan', 'Haveri', 'Kalaburagi', 'Kodagu', 'Kolar', 'Koppal', 'Mandya', 'Mysuru', 'Raichur', 'Ramanagara', 'Shivamogga', 'Tumakuru', 'Udupi', 'Uttar kannada', 'Vijayapura', 'Yadgir'],
        "KERALA": ['Alappuzha', 'Ernakulam', 'Idukki', 'Kannur', 'Kasargod', 'Kollam', 'Kottayam', 'Kozhikode', 'Malappuram', 'Palakkad', 'Pathanamthitta', 'Thiruvananthapuram', 'Thrissur', 'Wayanad'],
        "MADHYAPRADESH": ['Agar', 'Alirajpur', 'Anuppur', 'Ashoknagar', 'Balaghat', 'Barwani', 'Betul', 'Bhind', 'Bhopal', 'Burhanpur', 'Chhatarpur', 'Chhindwara', 'Damoh', 'Datia', 'Dewas', 'Dhar', 'Dindori', 'Guna', 'Gwalior', 'Harda', 'Hoshangabad', 'Indore', 'Jabalpur', 'Jhabua', 'Katni', 'Khandwa', 'Khargone', 'Mandla', 'Mandsaur', 'Morena', 'Narsinghpur', 'Neemuch', 'Panna', 'Raisen', 'Rajgarh', 'Ratlam', 'Rewa', 'Sagar','Satna','Sehore','Seoni','Shahdol','Shajapur','Sheopur','Shivpuri','Sidhi','Singrauli','Tikamgarh','Ujjain','Umaria','Vidisha'],
        "MAHARASHTRA": ['Ahmednagar', 'Akola', 'Amravati', 'Aurangabad', 'Beed', 'Bhandara', 'Buldhana', 'Chandrapur', 'Dhule', 'Gadchiroli', 'Gondia', 'Hingoli', 'Jalgaon', 'Jalna', 'Kolhapur', 'Latur', 'Nagpur', 'Nanded', 'Nandurbar', 'Nashik', 'Osmanabad', 'Palghar', 'Parbhani', 'Pune', 'Raigad', 'Ratnagiri', 'Sangli', 'Satara', 'Sindhudurg', 'Solapur', 'Thane', 'Wardha', 'Washim','Yawatmal'],
        "MANIPUR": ['Bishnupur', 'Chandel', 'Churachandpur', 'Imphal East', 'Jiribam', 'Imphal West', 'Senapati', 'Tamenglong', 'Thoubal', 'Ukhrul'],
        "MIZORAM": ['Aizawl', 'Champhai', 'Kolasib', 'Lawngtlai', 'Lunglei', 'Mamit', 'Saiha', 'Serchhip'],
        "NAGALAND": ['Dimapur', 'Kiphire', 'Kohima', 'Longleng', 'Mokokchung', 'Mon', 'Peren', 'Phek', 'Tuenchung', 'Wokha', 'Zunheboto'],
        "ODISHA": ['Angul', 'Balasore', 'Bargarh', 'Bhadrak', 'Bolangir', 'Boudh', 'Cuttack', 'Deogarh', 'Dhenkanal', 'Gajapati', 'Ganjam', 'Jagatsinghpur', 'Jajpur', 'Jharsuguda', 'Kalahandi', 'Kandhamal', 'Kendrapara', 'Keonjhar', 'Khurda', 'Koraput', 'Malkangiri', 'Mayurbhanj', 'Nabarangapur', 'Nayagarh', 'Nuapada', 'Puri', 'Rayagada', 'Sambalpur', 'Subarnapur', 'Sundargarh'],
        "PUNJAB": ['Amritsar', 'Barnala', 'Bathinda', 'Faridkot', 'Fatehgarh Sahib', 'Fazilka', 'Ferozepur', 'Gurdaspur', 'Hoshiarpur', 'Jalandhar', 'Kapurthala', 'Ludhiana', 'Mansa', 'Moga', 'Mohali', 'Muktsar', 'Nawanshahar', 'Pathankot', 'Patiala', 'Ropar', 'Sangrur', 'Tarn Taran'],
        "RAJASTHAN": ['Ajmer', 'Alwar', 'Banswara', 'Baran', 'Barmer', 'Bharatpur', 'Bhilwara', 'Bikaner', 'Bundi', 'Chittaurgarh', 'Churu', 'Dausa', 'Dhaulpur', 'Dungarpur', 'Ganganagar', 'Hanumangarh', 'Jaipur', 'Jaisalmer', 'Jalor', 'Jhalawar', 'Jhunjhunun', 'Jodhpur', 'Karauli', 'Kota', 'Nagaur', 'Pali', 'Pratapgarh', 'Rajsamand', 'Sawai Madhopur', 'Sikar', 'Sirohi', 'Tonk', 'Udaipur'],
        "SIKKIM": ['East District', 'South District', 'North District', 'West District'],
        "TAMIL NADU": ['Ariyalur', 'Chennai', 'Coimbatore', 'Cuddalore', 'Dharmapuri', 'Dindigul', 'Erode', 'Kancheepuram', 'Kanyakumari', 'Karur', 'Krishnagiri', 'Madurai', 'Nagapattinam', 'Namakkal', 'Nilgiris', 'Peramabalur', 'Pudukottai', 'Ramanathapuram', 'Salem', 'Sivagangai', 'Thanjavur', 'Theni', 'Thiruppur', 'Thiruvallur', 'Thoothukudi', 'Tirunelveli', 'Tiruvannamalai', 'Tiruvarur', 'Trichy', 'Vellore', 'Villupuram', 'Viruthunagar'],
        "TELANGANA": ['Adilabad', 'Bhadradri Kothagudem', 'Hyderabad', 'Jagtial', 'Jangaon', 'Jayashankar Bhupalapally', 'Jogulamba Gadwal', 'Kamareddy', 'Karimnagar', 'Khammam', 'Komarambhem Asifabad', 'Mahabubabad', 'Mahabubnagar', 'Mancherial', 'Medak', 'Medchal Malkajgiri', 'Nagarkurnool', 'Nalgonda', 'Nirmal', 'Nizamabad', 'Peddapalli', 'Rajanna Sircilla', 'Rangareddy', 'Sangareddy', 'Siddipet', 'Suryapet', 'Vikarabad', 'Wanaparthy', 'Warangal Rural', 'Warangal Urban', 'Yadadri Bhongiri'],
        "TRIPURA": ['Dhalai', 'Gomati', 'Khowai', 'North Tripura', 'Sepahijala', 'South Tripura', 'Unakoti', 'West Tripura'],
        "UTTAR PRADESH": ['Agra', 'Aligarh', 'Ambedkar Nagar', 'Amethi', 'Amroha', 'Auraiya', 'Ayodhya', 'Azamgarh', 'Bagpat', 'Bahraich', 'Ballia', 'Balrampur', 'Banda', 'Barabanki', 'Bareilly', 'Basti', 'Bijnor', 'Budaun', 'Bulandshahar', 'Chandauli', 'Chitrakoot', 'Deoria', 'Etah', 'Etawah', 'Farrukhabad', 'Fatehpur', 'Firozabad', 'G.B.Nagar', 'Ghaziabad', 'Ghazipur', 'Gonda', 'Gorakhpur', 'Hamirpur', 'Hapur', 'Hardoi', 'Hathras', 'Jalaun', 'Jaunpur', 'Jhansi', 'Kannauj','Kanpur Dehat','Kanpur Nagar','Kasganj','Kaushambi','Kushi Nagar','Lakhimpur Kheri','Lalitpur','Lucknow','Mahoba','Mahrajganj','Mainpuri','Mathura','Maunath Bhanjan','Meerut','Mirzapur','Moradabad','Muzaffarnagar','Pilibhit','Pratapgarh','Prayagraj','Raibareli','Rampur','Saharanpur','Sambhal','Sant Kabir Nagar','Sant Ravidas Nagar','Shahjahanpur','Shamli','Shrawasti','Siddharth Nagar','Sitapur','Sonbhadra','Sultanpur','Unnao','Varanasi'],
        "UTTARAKHAND": ['Almora', 'Bageshwar', 'Chamoli', 'Champawat', 'Dehradun', 'Haridwar', 'Nainital', 'Pauri Garhwal', 'Pithoragarh', 'Rudraprayag', 'Tehri Garhwal', 'Udham Singh Nagar', 'Uttarkashi'],
        "WEST BENGAL": ['Alipurduar', 'Bankura', 'Birbhum', 'Cooch Behar', 'Dakshin Dinajpur', 'Darjeeling', 'Hooghly', 'Howrah', 'Jalpaiguri', 'Jhargram', 'Kalimpong', 'Kolkata', 'Malda', 'Murshidabad', 'Nadia', 'North 24-Parganas', 'Paschim Medinipore', 'Purba Medinipore', 'Purulia', 'South 24-Parganas', 'Uttar Dinajpur'],
        "A&N": ['North & Middle Andaman', 'Nicobar', 'South Andaman'],
        "CHANDIGARH": ['UT of Chandigarh'],
        "Dadra & Nagar Haveli": ['Dadra & Nagar Haveli'],
        "Daman": ['Daman'],
        "Diu": ['Diu'],
        "LAKSHADWEEP": ['Agatti', 'Amini', 'Androth', 'Chetlat', 'Kadmat', 'Kalpeni', 'Kiltan', 'Kavaratti', 'Minicoy'],
        "PUDUCHERRY": ['Puducherry', 'Karaikal', 'Mahe', 'Yanam'],
        }

# Define input columns
col1, col2 = st.columns(2)  # Create two columns

with col1:
    name_of_state = st.selectbox('Select State', ['Enter state','ANDHRA PRADESH', 'ARUNACHAL PRADESH', 'ASSAM','BIHAR','CHATTISGARH','DELHI','GOA','GUJARAT','HARYANA','HIMACHAL','J&K','JHARKHAND','KARNATAKA','KERALA','MADHYAPRADESH','MAHARASHTRA','MANIPUR','MEGHALAY','MIZORAM','NAGALAND','ODISHA','PUNJAB','RAJASTHAN','SIKKIM','TAMILNADU','TELANGANA','TRIPURA','UTTARPRADESH','WEST BENGAL','A&N ISLAND','CHANDIGARH','Dadra & Nagar Haveli','Daman','Diu','LAKSHADWEEP','Puducherry'])
    selected_district = st.selectbox('Select District',
                                     districts.get(name_of_state, ['No district available for selected state']))
    recharge_from_rainfall_monsoon = st.text_input('Recharge from rainfall During Monsoon Season')
    recharge_from_other_sources_monsoon = st.text_input('Recharge from other sources During Monsoon Season')

with col2:
    recharge_from_rainfall_non_monsoon = st.text_input('Recharge from rainfall During Non Monsoon Season')
    recharge_from_other_sources_non_monsoon = st.text_input('Recharge from other sources During Non Monsoon Season')
    total_natural_discharges = st.text_input('Total Natural Discharges')
    

# Button to trigger prediction
if st.button('Total Extractable Ground Water resource'):
    # Check if all input fields are filled
    if not recharge_from_rainfall_monsoon or not recharge_from_other_sources_monsoon or not recharge_from_rainfall_non_monsoon or not recharge_from_other_sources_non_monsoon or not total_natural_discharges:
        st.warning("Please enter values for all input fields.")
    else:
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Recharge from rainfall During Monsoon Season': [float(recharge_from_rainfall_monsoon)],
            'Recharge from other sources During Monsoon Season': [float(recharge_from_other_sources_monsoon)],
            'Recharge from rainfall During Non Monsoon Season': [float(recharge_from_rainfall_non_monsoon)],
            'Recharge from other sources During Non Monsoon Season': [float(recharge_from_other_sources_non_monsoon)],
            'Total Natural Discharges': [float(total_natural_discharges)],
        })

        # Make predictions using the loaded model
        water_level_prediction = model.predict(input_data)

        # Display results
        st.success(f"Total Extractable Ground Water resource: {water_level_prediction[0]:.2f} bcm")

        filtered_df = df[(df['Name of State'] == name_of_state) & (df['Name of District'] == selected_district)]

        if filtered_df.empty:
            st.info(f"Data not found for the given state and district {name_of_state} and {selected_district}.")
        else:
            # Get the total current annual ground water extraction
            total_extraction = filtered_df['Total Current Annual Ground Water Extraction'].values[0]
            st.info(f"Total Current Annual Ground Water Extraction for {name_of_state}, {selected_district}: {total_extraction:.2f} bcm")

            # Calculate the stage of ground water extraction
            stage_extraction = (total_extraction / water_level_prediction[0]) * 100
            st.info(f"Stage of Ground Water extraction: {stage_extraction:.2f}%")
        # Display min and max water levels for the selected state
    min_level = water_level_df.loc[water_level_df['State'] == name_of_state, 'min'].values
    max_level = water_level_df.loc[water_level_df['State'] == name_of_state, 'max'].values
    if min_level and max_level:
        st.info(f"Minimum water level: {min_level[0]} m b.g.l")
        st.info(f"Maximum water level: {max_level[0]} m b.g.l")
    else:
        st.warning("Water level data not available for the selected state.")