using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking;

public class WebClient : MonoBehaviour
{
    IEnumerator SendData(string data)
    {
        WWWForm form = new WWWForm();
        form.AddField("bundle", "the data");
        string url = "http://127.0.0.1:5000"; // Actualiza la URL al servidor correcto
        using (UnityWebRequest www = UnityWebRequest.Post(url, form))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(data);
            www.uploadHandler = (UploadHandler)new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.isNetworkError || www.isHttpError)
            {
                Debug.Log(www.error);
            }
            else
            {
                string jsonResponse = www.downloadHandler.text;
                SimulationData simData = JsonUtility.FromJson<SimulationData>(jsonResponse);

                UpdateSimulation(simData);
            }
        }
    }

    [System.Serializable]
    public class SimulationData
    {
        public List<int[]> paredes;
        public List<int[]> puertas;
        public List<int[]> entradas;
        public List<int[]> fuegos;
        public int num_agents;
        public int largo;
        public int ancho;
        public string resultados_simulacion;
    }

    void UpdateSimulation(SimulationData data)
    {
        // Actualiza los objetos y el estado en Unity basado en 'data'
        Debug.Log("Datos recibidos y aplicados en Unity.");
    }

    void Start()
    {
        Vector3 fakePos = new Vector3(3.44f, 0, -15.707f);
        string json = EditorJsonUtility.ToJson(fakePos);
        StartCoroutine(SendData(json));
    }

    void Update()
    {
        
    }
}

