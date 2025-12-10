// Gemini API Service for Report Generation

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY || ''
const API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent'

export interface ModelPerformance {
  name: string
  r2: number
  rmse: number
  mae: number
  mape: number
}

export const modelData: ModelPerformance[] = [
  { name: 'Ensemble', r2: 0.7705, rmse: 1.8011, mae: 0.4582, mape: 0.6380 },
  { name: 'Linear Regression', r2: 0.7702, rmse: 1.8024, mae: 0.4486, mape: 0.6224 },
  { name: 'Neural Network', r2: 0.7547, rmse: 1.8622, mae: 0.5938, mape: 0.8373 },
  { name: 'XGBoost', r2: 0.7036, rmse: 2.0469, mae: 0.8443, mape: 1.2136 },
  { name: 'Random Forest', r2: 0.6869, rmse: 2.1038, mae: 1.0465, mape: 1.5216 },
]

export const featureImportance = [
  'Attendance (35.6%)',
  'Hours_Studied (10.4%)',
  'Access_to_Resources (7.9%)',
  'Parental_Involvement (7.2%)',
  'Previous_Scores (5.5%)',
  'Tutoring_Sessions (4.0%)',
  'Peer_Influence (3.6%)',
  'Family_Income (3.5%)',
]

export const datasetInfo = {
  totalSamples: 6607,
  trainingSamples: 5285,
  testSamples: 1322,
  features: 20,
  target: 'Exam_Score',
  targetRange: '55-101',
  targetMean: 67.24,
}

export type Language = 'en' | 'zh-TW'

export const languageLabels: Record<Language, string> = {
  'en': 'English',
  'zh-TW': '繁體中文',
}

const getPromptForReportType = (reportType: string, language: Language = 'en'): string => {
  const languageInstruction = language === 'zh-TW' 
    ? '\n\n**IMPORTANT: Generate the entire response in Traditional Chinese (繁體中文). Use Traditional Chinese characters only.**\n'
    : '\n\n**Generate the response in English.**\n'

  const context = `
You are analyzing a Machine Learning project for predicting student exam performance.

Dataset Information:
- Total samples: ${datasetInfo.totalSamples}
- Training samples: ${datasetInfo.trainingSamples}
- Test samples: ${datasetInfo.testSamples}
- Number of features: ${datasetInfo.features}
- Target variable: ${datasetInfo.target} (range: ${datasetInfo.targetRange}, mean: ${datasetInfo.targetMean})

Model Performance Results:
${modelData.map(m => `- ${m.name}: R²=${m.r2.toFixed(4)}, RMSE=${m.rmse.toFixed(4)}, MAE=${m.mae.toFixed(4)}, MAPE=${m.mape.toFixed(4)}%`).join('\n')}

Top Feature Importance (from XGBoost analysis):
${featureImportance.join('\n')}

Key Findings:
- Attendance is the most important predictor (35.6% importance)
- The Ensemble model achieves the best overall performance
- Linear Regression surprisingly performs nearly as well as the Ensemble
- All models achieve R² > 0.68, indicating good predictive power
${languageInstruction}`

  const prompts: Record<string, string> = {
    overall: `${context}

Generate a comprehensive overall performance report for this Student Performance Prediction ML project. Include:
1. Executive Summary (2-3 sentences)
2. Model Performance Analysis (compare all models)
3. Key Insights from the data
4. Strengths of our approach
5. Conclusion

Format with clear headings using markdown. Be professional but accessible. Keep it concise but informative.`,

    'by-model': `${context}

Generate a detailed report analyzing each model individually. For each model (Ensemble, Linear Regression, Neural Network, XGBoost, Random Forest), provide:
1. Brief description of the algorithm
2. Performance metrics interpretation
3. Why it performed the way it did
4. When to use this model

Format with clear headings using markdown. Be analytical and educational.`,

    students: `${context}

Based on the feature importance and model analysis, generate actionable recommendations for STUDENTS to improve their academic performance. Include:
1. Top 5 actionable strategies based on what the data shows matters most
2. Study habits recommendations (Hours_Studied matters)
3. Attendance importance (most crucial factor!)
4. How to leverage available resources
5. Motivation and peer influence tips

Format with clear headings using markdown. Be encouraging and practical. Make it relevant to students.`,

    parents: `${context}

Based on the feature importance and model analysis, generate recommendations for PARENTS to support their children's academic success. Include:
1. The importance of parental involvement (key factor in the data)
2. How to provide better access to resources
3. Supporting healthy sleep and physical activity habits
4. Creating a conducive learning environment
5. Working with teachers and schools

Format with clear headings using markdown. Be supportive and practical.`,

    teaching: `${context}

Based on the feature importance and model analysis, generate recommendations for TEACHING UNITS (schools, educators, policy makers). Include:
1. Evidence-based insights from the data
2. Importance of attendance policies and engagement
3. Teacher quality impact
4. Resource allocation recommendations
5. Early intervention strategies using predictive models
6. How to use this ML system for student support

Format with clear headings using markdown. Be professional and evidence-based.`,
  }

  return prompts[reportType] || prompts.overall
}

export async function* generateReportStream(reportType: string, language: Language = 'en'): AsyncGenerator<string, void, unknown> {
  const prompt = getPromptForReportType(reportType, language)

  try {
    const response = await fetch(`${API_URL}?key=${API_KEY}&alt=sse`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [{ text: prompt }],
          },
        ],
        generationConfig: {
          temperature: 0.7,
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 2048,
        },
      }),
    })

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No response body')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6)
          if (jsonStr.trim() === '') continue
          
          try {
            const data = JSON.parse(jsonStr)
            const text = data.candidates?.[0]?.content?.parts?.[0]?.text
            if (text) {
              yield text
            }
          } catch {
            // Skip invalid JSON
          }
        }
      }
    }
  } catch (error) {
    console.error('Gemini API Error:', error)
    throw error
  }
}

// Fallback non-streaming version
export async function generateReport(reportType: string, language: Language = 'en'): Promise<string> {
  const prompt = getPromptForReportType(reportType, language)
  const nonStreamUrl = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'

  try {
    const response = await fetch(`${nonStreamUrl}?key=${API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [{ text: prompt }],
          },
        ],
        generationConfig: {
          temperature: 0.7,
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 2048,
        },
      }),
    })

    if (!response.ok) {
      const errorData = await response.text()
      throw new Error(`API request failed: ${response.status} - ${errorData}`)
    }

    const data = await response.json()
    return data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response generated'
  } catch (error) {
    console.error('Gemini API Error:', error)
    throw error
  }
}

