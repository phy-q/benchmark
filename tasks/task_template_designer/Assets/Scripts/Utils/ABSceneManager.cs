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

﻿using UnityEngine;
using System.Collections;
using UnityEngine.SceneManagement;

public class ABSceneManager : ABSingleton<ABSceneManager> {

	public delegate void ActionBetweenScenes();

	public AudioClip _backgroundMusic;

	private string _lastSceneName;
	public string LastSceneName { get { return _lastSceneName; } }

	private int _lastScene;
	public int LastScene { get { return _lastScene; } }

	private string _currentSceneName;
	public string CurrentSceneName { get { return _currentSceneName; } }

	private int _currentScene;
	public int CurrentScene { get { return _currentScene; } }

	void Start() {

		_backgroundMusic = Resources.Load("Audio/title_theme") as AudioClip;
	}
			
	public void LoadScene (string sceneName, bool showLoadingScreen = true, ActionBetweenScenes actioneBetweenScenes = null) {

		ABSceneManager.Instance.StartCoroutine(SceneSwitchCoroutine(sceneName, showLoadingScreen, actioneBetweenScenes));
	}

	IEnumerator SceneSwitchCoroutine (string sceneName, bool showLoadingScreen, ActionBetweenScenes actioneBetweenScenes) {

		_lastScene = SceneManager.GetActiveScene ().buildIndex;
		_lastSceneName = SceneManager.GetActiveScene ().name;

		if(showLoadingScreen) 
			SceneManager.LoadScene("LoadingScene");

		yield return new WaitForSeconds(0.1f);

		if (actioneBetweenScenes != null)
			actioneBetweenScenes();

		SceneManager.LoadScene(sceneName);

		if (sceneName.EndsWith("Menu")) {
			if(!ABAudioController.Instance.IsPlayingMusic(_backgroundMusic))
				ABAudioController.Instance.PlayMusic(_backgroundMusic);
		}
		else
			ABAudioController.Instance.StopMusic();

		_currentScene = SceneManager.GetActiveScene ().buildIndex;
		_currentSceneName = SceneManager.GetActiveScene ().name;
	}
}

