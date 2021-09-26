using UnityEngine;
using System.Collections;

public class ABBirdYellow : ABBird {

	public float _specialAttackForce = 2f;

	void SpecialAttack() {
		Vector2 force = _rigidBody.velocity.normalized * _specialAttackForce;
		_rigidBody.AddForce(force, ForceMode2D.Impulse);
	}
}
