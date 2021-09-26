using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ABBirdDown : ABBird
{
	int _specialAttackForce = 10;
	void SpecialAttack()
	{
		_rigidBody.velocity = new Vector2(0,0);
		Vector2 force = new Vector2(0, -1) * _specialAttackForce;
		_rigidBody.AddForce(force, ForceMode2D.Impulse);

	}
}

