using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// A singleton with which a gameObject holds all the ABSingletons
/// </summary>
public class ABGameManager : MonoBehaviour
{
    public static ABGameManager Instance;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(this);
        }
        else
        {
            Destroy(gameObject);
        }
    }
}
