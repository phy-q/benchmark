// SCIENCE BIRDS: A clone version of the Angry Birds game used for 
// research purposes
// 
// Copyright (C) 2016 - Lucas N. Ferreira - lucasnfe@gmail.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>
//

using UnityEngine;
using System.Collections;

public class ABSingleton<T> : MonoBehaviour where T : MonoBehaviour
{

    //Here is a private reference only this class can access
    private static T _instance;

    // Avoid calling contructor with new
    protected ABSingleton()
    { }

    //This is the public reference that other classes will use
    public static T Instance
    {
        get
        {
            //If _instance hasn't been set yet, we grab it from the scene!
            //This will only happen the first time this reference is used.
            if (_instance == null)
                _instance = (T)FindObjectOfType(typeof(T));

            if (_instance == null)
            {
                _instance = ABGameManager.Instance.gameObject.AddComponent<T>();
            }

            return _instance;
        }
    }
}
