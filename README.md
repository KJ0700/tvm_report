# tvm_report

# AI Performance Analytics Dashboard

A comprehensive Streamlit-based dashboard for analyzing AI chatbot performance with detailed metrics, visualizations, and insights.

## 📋 Overview

This interactive dashboard provides in-depth analysis of AI chatbot interactions, including:
- Performance metrics and KPIs
- Cost and token usage analysis
- Intent distribution and categorization
- Entity extraction performance
- Dealer-wise comparison
- Temporal patterns
- Topic modeling for user messages

## 🚀 Features

### 1. Executive Overview
- Total AI calls and interactions tracking
- Session metrics and actionable rate
- Date range analysis
- Unique dealers and intent types

### 2. Cost & Token Analysis
- Detailed breakdown by dealer
- Total cost calculations
- Input/output token tracking
- Per-dealer metrics with visualizations

### 3. Intent Category Analysis
- Categorization into Actionable, FAQ, General, and Irrelevant
- Visual pie charts
- Category-wise statistics

### 4. Detailed Intent Analysis
- All intents with count and percentage
- Top 5 intents visualization
- Comprehensive intent statistics table

### 5. Entity Extraction Performance
- Current vehicle information
- Interested vehicle details
- Personal information (name, email, phone, DOB, address)
- Appointment scheduling
- Service information
- Finance details
- Intent-level extraction rates

### 6. Dealer Performance Comparison
- Interactions and sessions per dealer
- Actionable rate comparison
- Average messages per session
- Category distribution (General, FAQ, Irrelevant)

### 7. Temporal Patterns
- Hourly interaction distribution
- Day-of-week analysis
- Date-wise trends

### 8. Topic Modeling (Optional)
- BERTopic-based analysis for GSL, GSR, IR, and FAQ intents
- Enhanced text preprocessing
- Topic visualization and keywords
- Sample messages per topic

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Step 1: Clone or Download the Repository

```bash
cd data_analysis_V2
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

For basic dashboard functionality:
```bash
pip install -r requirements_streamlit.txt
```

For topic modeling features (optional):
```bash
pip install -r requirements_topic_modeling.txt
```

**Note for Windows users:** If you encounter issues installing HDBSCAN, refer to `INSTALL_HDBSCAN_WINDOWS.md`

## 🎯 Usage

### Running the Dashboard

1. Activate your virtual environment (if not already activated)
2. Run the Streamlit application:

```bash
streamlit run streamlit_report.py
```

3. The dashboard will open in your default web browser (typically at `http://localhost:8501`)

### Uploading Data

1. Click "Browse files" in the sidebar
2. Upload your CSV file containing AI chatbot interaction data
3. The dashboard will automatically analyze and display the results

## 📊 Data Format

The CSV file should contain the following columns:

### Required Columns:
- `sessionId`: Unique identifier for each conversation session
- `dealerId`: Dealer identification number
- `intentName`: Intent classification (e.g., FV, GSL, GSR, IR, BV, etc.)
- `userMessage`: The user's input message
- `aiResponse_json`: JSON response from AI with extracted entities
- `dateCreatedUtc`: Timestamp in UTC format
- `promptName`: Prompt type (e.g., initial, FAQ, secondary)

### Optional Columns (for cost analysis):
- `pricePerCall`: Cost per API call
- `inputTokenCount`: Number of input tokens
- `outputTokenCount`: Number of output tokens
- `latencyMs`: Response latency in milliseconds

### Sample aiResponse_json Structure:

```json
{
  "intentCode": "FV",
  "intentDescription": "Find Vehicle",
  "interestedMake": "Toyota",
  "interestedModel": "Camry",
  "interestedYear": "2024",
  "firstName": "John",
  "phoneNumber": "1234567890"
}
```

## 🎨 Features Explained

### Data Filtering

The dashboard automatically filters out certain combinations:
- **IR + initial**: Irrelevant intents from initial prompts
- **IR + FAQ**: Irrelevant intents from FAQ prompts

This ensures cleaner metrics for Total Interactions while maintaining Total AI Calls for cost calculations.

### Intent Categories

1. **Actionable**: SS, STD, VYT, SYC, BV, GPQ, MDS, SC, OPT, FV, DS, TXT
2. **FAQ**: FAQ, faq
3. **General**: GSL, GSR
4. **Irrelevant**: IR

### Entity Extraction Categories

1. **Current Vehicle**: currentYear, currentMake, currentModel, currentTrim
2. **Interested Vehicle**: interestedYear, interestedMake, interestedModel, interestedTrim, color, interiorColor, bodyType
3. **Personal Info**: firstName, lastName, middleName, email, phoneNumber, address, zipCode, city, dateOfBirth
4. **Appointments**: appointmentDate, appointmentTime
5. **Service Info**: serviceTypes, mileage
6. **Finance Info**: downPayment, mileageAllowance, term, financeType

## 📝 Report Sections

Use the sidebar checkboxes to toggle different sections:
- ✅ Executive Overview
- ✅ Cost & Token Analysis
- ✅ Intent Analysis
- ✅ Category Analysis
- ✅ Entity Extraction
- ✅ Dealer Comparison
- ✅ Temporal Patterns
- ⬜ Topic Modeling (requires optional dependencies)

## 🛠️ Troubleshooting

### Common Issues

**1. Package Installation Errors**
- Ensure you're using Python 3.11+
- Try upgrading pip: `pip install --upgrade pip`
- For Windows HDBSCAN issues, see `INSTALL_HDBSCAN_WINDOWS.md`

**2. CSV Upload Fails**
- Verify all required columns are present
- Check for proper JSON formatting in `aiResponse_json`
- Ensure date format is compatible

**3. Topic Modeling Not Available**
- Install optional dependencies: `pip install -r requirements_topic_modeling.txt`
- Restart the Streamlit application

**4. Empty or Zero Metrics**
- Verify data filtering isn't removing all records
- Check that intentName and promptName columns have expected values
- Ensure aiResponse_json contains valid JSON

## 📚 Additional Documentation

- `TOPIC_MODELING_README.md`: Detailed guide on topic modeling features
- `TOPIC_MODELING_EXPLANATION.md`: Technical explanation of the topic modeling approach
- `TOPIC_MODELING_QUICK_REFERENCE.md`: Quick reference for topic modeling
- `TOPIC_MODELING_PRESENTATION.md`: Presentation-ready overview
- `INSTALL_HDBSCAN_WINDOWS.md`: Windows-specific installation guide for HDBSCAN

## 🔧 Configuration

The dashboard uses default settings that work for most use cases. To modify:

1. **Intent Categories**: Edit the `INTENT_CATEGORIES` dictionary in `streamlit_report.py`
2. **Entity Fields**: Modify the field lists in the `get_entity_extraction_analysis()` method
3. **Chart Colors**: Adjust color schemes in the visualization functions

## 📈 Performance Tips

- For large datasets (>10,000 rows), topic modeling may take several minutes
- Disable unused sections via sidebar to improve load times
- Use the filtering features to focus on specific date ranges or dealers

## 🤝 Support

For issues, questions, or feature requests, please refer to the documentation files or contact the development team.

## 📄 License

[Add your license information here]

## 🔄 Version History

- **v2.0**: Added Total AI Calls tracking, improved filtering logic, enhanced entity extraction
- **v1.0**: Initial release with basic analytics and topic modeling

---

**Last Updated:** February 11, 2026
