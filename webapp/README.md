# Hemolysis JSP GUI

This folder provides a JSP-based GUI for uploading amino-acid sequence CSV files
and showing hemolysis prediction results.

## Files

- `hemolysis.jsp`: Upload page and result table.
- `WEB-INF/web.xml`: Sets `hemolysis.jsp` as welcome page.

## Expected backend API

The page sends a `POST` request to:

- `${contextPath}/api/hemolysis/predict`

With multipart form field:

- `file`: CSV file.

Expected JSON response example:

```json
{
  "results": [
    {
      "id": "pep_001",
      "sequence": "GLFDIVKKVVGAFGSL",
      "prediction": "hemolytic",
      "confidence": 0.92
    }
  ]
}
```
